import json
import os
import re
import hashlib
import io
import hmac
import secrets
import sqlite3
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
import zlib
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from google import genai

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
google_client_id = os.getenv("GOOGLE_CLIENT_ID")

AUTH_DATABASE = Path(os.getenv("AUTH_DATABASE_PATH", Path(__file__).with_name("byteforce.sqlite3")))
PASSWORD_HASH_ROUNDS = 600_000
ACCESS_TOKEN_SECONDS = 30 * 60
REFRESH_TOKEN_SECONDS = 14 * 24 * 60 * 60


def auth_db():
    AUTH_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(AUTH_DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def auth_connection():
    connection = auth_db()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_auth_database():
    with auth_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'candidate',
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                access_token_hash TEXT NOT NULL,
                refresh_token_hash TEXT NOT NULL,
                access_expires_at INTEGER NOT NULL,
                refresh_expires_at INTEGER NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_access ON auth_sessions(access_token_hash);
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_refresh ON auth_sessions(refresh_token_hash);
            CREATE TABLE IF NOT EXISTS interview_reports (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                company TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                score INTEGER NOT NULL,
                questions_answered INTEGER NOT NULL,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                report_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_interview_reports_user_date ON interview_reports(user_id, created_at DESC);
            """
        )


init_auth_database()


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Enter your name.")
        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address.")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class GoogleSignInRequest(BaseModel):
    credential: str = Field(min_length=20, max_length=10000)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=256)


class UserPublic(BaseModel):
    id: int
    name: str
    email: str
    role: str


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)


class InterviewStartRequest(BaseModel):
    role: str = Field(min_length=1, max_length=160)
    domain: str = Field(default="General", max_length=100)
    skills: str = Field(default="", max_length=1000)
    company: str = Field(default="General practice", max_length=160)
    interview_type: str = Field(default="Mock interview", max_length=100)
    difficulty: str = Field(default="Intermediate", max_length=40)
    resume_context: str = Field(default="", max_length=12000)
    language: str = Field(default="English", max_length=40)


class InterviewExchange(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    answer: str = Field(min_length=1, max_length=12000)
    score: int = Field(default=0, ge=0, le=100)
    feedback: str = Field(default="", max_length=3000)
    strengths: list[str] = Field(default_factory=list, max_length=8)
    improvements: list[str] = Field(default_factory=list, max_length=8)


class InterviewTurnRequest(InterviewStartRequest):
    question: str = Field(min_length=1, max_length=3000)
    answer: str = Field(min_length=1, max_length=12000)
    history: list[InterviewExchange] = Field(default_factory=list, max_length=12)
    turn_number: int = Field(ge=1, le=12)
    max_turns: int = Field(default=6, ge=1, le=12)


class ResumeTextRequest(BaseModel):
    text: str = Field(min_length=80, max_length=20000)


class InterviewReportRequest(InterviewStartRequest):
    history: list[InterviewExchange] = Field(min_length=1, max_length=12)
    duration_seconds: int = Field(default=0, ge=0, le=86400)
    focus_changes: int = Field(default=0, ge=0, le=100)



ROLE_SKILLS = {
    "Frontend Developer": ["frontend developer", "front end", "frontend", "react", "javascript", "typescript", "html", "css", "vite", "next.js", "angular", "vue"],
    "Backend Developer": ["python", "java", "node.js", "fastapi", "django", "spring", "api", "microservices", "sql"],
    "Full Stack Developer": ["full stack", "frontend", "backend", "react", "node.js", "javascript", "typescript"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "python", "data analysis", "statistics"],
    "DevOps Engineer": ["aws", "azure", "docker", "kubernetes", "ci/cd", "terraform", "linux"],
    "Product Manager": ["product roadmap", "product management", "stakeholder", "user research", "product strategy"],
    "Software Engineer": ["software engineer", "software development", "algorithms", "data structures", "git"],
}


def resume_profile(text: str) -> dict[str, Any]:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email removed]", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d .()/-]{8,}\d)(?!\w)", "[phone removed]", text)
    text = re.sub(r"\s+", " ", text).strip()[:12000]
    lower = text.lower()
    terms = {skill for skills in ROLE_SKILLS.values() for skill in skills}
    def contains(term: str) -> bool:
        return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", lower) is not None
    skills = sorted(skill for skill in terms if contains(skill))
    scores = {role: sum(1 for skill in items if contains(skill)) for role, items in ROLE_SKILLS.items()}
    suggested_role = max(scores, key=scores.get)
    if scores[suggested_role] == 0:
        suggested_role = "Software Engineer"
    return {"text": text, "suggested_role": suggested_role, "skills": skills[:20]}


def extract_pdf_text(content: bytes) -> str:
    pieces = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", content, re.DOTALL):
        stream = match.group(1)
        dictionary = content[max(0, match.start() - 300):match.start()]
        if b"/FlateDecode" in dictionary:
            try:
                stream = zlib.decompress(stream)
            except zlib.error:
                continue
        for raw in re.findall(rb"\(((?:\\.|[^\\)])*)\)\s*(?:Tj|')", stream):
            raw = re.sub(rb"\\([nrtbf()\\])", lambda m: {b"n":b"\n",b"r":b"\r",b"t":b"\t",b"b":b"\b",b"f":b"\f",b"(":b"(",b")":b")",b"\\":b"\\"}[m.group(1)], raw)
            raw = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8)]), raw)
            pieces.append(raw.decode("latin-1", errors="ignore"))
        for array in re.findall(rb"\[((?:\\.|[^\]])*)\]\s*TJ", stream):
            pieces.extend(raw.decode("latin-1", errors="ignore") for raw in re.findall(rb"\(((?:\\.|[^\\)])*)\)", array))
    return " ".join(pieces)


def parse_resume_bytes(filename: str, content: bytes) -> dict[str, Any]:
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Resume must be 10 MB or smaller.")
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".pdf":
            text = extract_pdf_text(content)
        elif suffix == ".docx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                xml = ET.fromstring(archive.read("word/document.xml"))
            text = " ".join(node.text or "" for node in xml.iter() if node.tag.endswith("}t"))
        elif suffix in {".txt", ".text"}:
            text = content.decode("utf-8-sig", errors="replace")
        else:
            raise HTTPException(status_code=415, detail="Upload a PDF, DOCX, or TXT resume.")
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise HTTPException(status_code=400, detail="The resume file could not be read. Try another PDF, DOCX, or TXT file.") from exc
    profile = resume_profile(text)
    if len(profile["text"]) < 40:
        raise HTTPException(status_code=422, detail="No selectable resume text was found. Try a text-based PDF or DOCX file.")
    return profile


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ROUNDS)
    return salt.hex(), digest.hex()


def public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}


def create_auth_session(user_id: int, user: dict[str, Any]) -> dict[str, Any]:
    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(48)
    now = int(time.time())
    with auth_connection() as connection:
        connection.execute(
            "INSERT INTO auth_sessions (id,user_id,access_token_hash,refresh_token_hash,access_expires_at,refresh_expires_at) VALUES (?,?,?,?,?,?)",
            (uuid.uuid4().hex, user_id, token_hash(access_token), token_hash(refresh_token), now + ACCESS_TOKEN_SECONDS, now + REFRESH_TOKEN_SECONDS),
        )
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_SECONDS, "user": user}


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Sign in to continue.", headers={"WWW-Authenticate": "Bearer"})
    with auth_connection() as connection:
        row = connection.execute(
            "SELECT u.id,u.name,u.email,u.role,s.id AS session_id FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.access_token_hash=? AND s.access_expires_at>? AND s.revoked=0",
            (token_hash(token), int(time.time())),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.", headers={"WWW-Authenticate": "Bearer"})
    return {**public_user(row), "session_id": row["session_id"]}


def require_role(*allowed_roles: str):
    def check_role(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="You do not have permission to use this resource.")
        return user
    return check_role


def require_client():
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing from the backend .env file",
        )


def gemini_json(prompt: str) -> dict[str, Any]:
    require_client()
    try:
        result = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        output = (result.text or "").strip()
        output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.IGNORECASE)
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini response did not contain a JSON object")
        data = json.loads(output[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("Gemini response must be a JSON object")
        return data
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini request failed or returned an unreadable response. Check the backend terminal for details.",
        ) from exc


@app.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(request: RegisterRequest):
    salt, digest = hash_password(request.password)
    try:
        with auth_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO users (name,email,password_salt,password_hash,created_at) VALUES (?,?,?,?,?)",
                (request.name.strip(), request.email, salt, digest, int(time.time())),
            )
            user_id = cursor.lastrowid
            user = {"id": user_id, "name": request.name.strip(), "email": request.email, "role": "candidate"}
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    return create_auth_session(user_id, user)


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):
    with auth_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE email=?", (request.email,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    _, candidate = hash_password(request.password, bytes.fromhex(row["password_salt"]))
    if not hmac.compare_digest(candidate, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    return create_auth_session(row["id"], public_user(row))


@app.post("/auth/google", response_model=AuthResponse)
def google_sign_in(request: GoogleSignInRequest):
    if not google_client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured. Set GOOGLE_CLIENT_ID in the backend and VITE_GOOGLE_CLIENT_ID in the frontend.")
    query = urllib.parse.urlencode({"id_token": request.credential})
    try:
        with urllib.request.urlopen(f"https://oauth2.googleapis.com/tokeninfo?{query}", timeout=8) as response:
            claims = json.loads(response.read(20000))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Google could not verify this sign-in. Please try again.") from exc
    email = str(claims.get("email", "")).strip().lower()
    name = str(claims.get("name") or claims.get("given_name") or email.split("@")[0]).strip()[:160]
    if claims.get("aud") != google_client_id or claims.get("email_verified") not in (True, "true") or not email:
        raise HTTPException(status_code=401, detail="Google account verification failed. Please try again.")
    with auth_connection() as connection:
        row = connection.execute("SELECT id,name,email,role FROM users WHERE email=?", (email,)).fetchone()
        if row:
            user_id = row["id"]
            user = public_user(row)
        else:
            salt, digest = hash_password(secrets.token_urlsafe(40))
            cursor = connection.execute(
                "INSERT INTO users (name,email,password_salt,password_hash,created_at) VALUES (?,?,?,?,?)",
                (name, email, salt, digest, int(time.time())),
            )
            user_id = cursor.lastrowid
            user = {"id": user_id, "name": name, "email": email, "role": "candidate"}
    return create_auth_session(user_id, user)


@app.post("/auth/refresh", response_model=AuthResponse)
def refresh_session(request: RefreshRequest):
    now = int(time.time())
    with auth_connection() as connection:
        row = connection.execute(
            "SELECT s.id AS session_id,u.id AS user_id,u.name,u.email,u.role FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.refresh_token_hash=? AND s.refresh_expires_at>? AND s.revoked=0",
            (token_hash(request.refresh_token), now),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Refresh token is invalid or expired.")
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(48)
        connection.execute(
            "UPDATE auth_sessions SET access_token_hash=?,refresh_token_hash=?,access_expires_at=?,refresh_expires_at=? WHERE id=?",
            (token_hash(access_token), token_hash(refresh_token), now + ACCESS_TOKEN_SECONDS, now + REFRESH_TOKEN_SECONDS, row["session_id"]),
        )
        user = {"id": row["user_id"], "name": row["name"], "email": row["email"], "role": row["role"]}
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_SECONDS, "user": user}


@app.post("/auth/logout")
def logout(user: dict[str, Any] = Depends(get_current_user)):
    with auth_connection() as connection:
        connection.execute("UPDATE auth_sessions SET revoked=1 WHERE id=?", (user["session_id"],))
    return {"message": "Signed out."}


@app.get("/auth/me", response_model=UserPublic)
def current_account(user: dict[str, Any] = Depends(get_current_user)):
    return user


@app.get("/auth/admin/users", response_model=list[UserPublic])
def list_accounts(user: dict[str, Any] = Depends(require_role("admin"))):
    with auth_connection() as connection:
        rows = connection.execute("SELECT id,name,email,role FROM users ORDER BY created_at DESC").fetchall()
    return [public_user(row) for row in rows]


@app.get("/")
def home():
    return {"message": "Hello, ByteForce! 🚀"}


@app.post("/ask")
def ask_gemini(request: AskRequest, user: dict[str, Any] = Depends(get_current_user)):
    require_client()
    try:
        result = client.models.generate_content(model="gemini-3.8-flash", contents=request.prompt)
        return {"answer": result.text or ""}
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini request failed. Check the backend terminal for details.",
        ) from exc


@app.post("/interview/start")
def start_interview(request: InterviewStartRequest, user: dict[str, Any] = Depends(get_current_user)):
    data = gemini_json(
        "Create the opening question for a realistic interview. Return only JSON with one key: "
        '{"question":"..."}. Keep it concise, relevant to the role and interview type, and calibrated '
        f"to difficulty. Role: {request.role}. Company: {request.company}. "
        f"Interview type: {request.interview_type}. Difficulty: {request.difficulty}. "
        f"Domain: {request.domain}. Candidate skills: {request.skills or 'not provided'}. "
        f" Write the question entirely in {request.language}; do not switch to English unless that is the selected language."
    )
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=502, detail="Gemini did not return an opening question.")
    return {"question": question.strip()}


@app.post("/interview/turn")
def interview_turn(request: InterviewTurnRequest, user: dict[str, Any] = Depends(get_current_user)):
    recent_history = request.history[-3:]
    first_recent_number = len(request.history) - len(recent_history) + 1
    prior = "\n\n".join(
        f"Question {index}: {item.question[:500]}\nCandidate answer: {item.answer[:1800]}\nEarlier feedback: {item.feedback[:500]}"
        for index, item in enumerate(recent_history, start=first_recent_number)
    ) or "No earlier answers."
    data = gemini_json(
        "You are a fair, encouraging interview coach. Evaluate only the answer provided; do not invent candidate evidence. "
        "Return only JSON with this exact shape: "
        '{"score":0,"feedback":"1-2 short sentences: one strength and one specific next step",'
        '"strengths":["..."],"improvements":["..."],"next_question":"...", "is_complete":false}. '
        "The score must be an integer from 0 to 100. Ask one relevant follow-up that adapts to the answer, role, difficulty, "
        "and prior conversation. Write the feedback and next question entirely in the selected language. "
        "Set is_complete true only when another question is unnecessary or this is the final turn; "
        "if complete, next_question may be an empty string. Keep feedback brief and specific. Limit the next question to one short sentence.\n\n"
        f"Role: {request.role}\nDomain: {request.domain}\nCandidate skills: {request.skills or 'not provided'}\n"
        f"Company: {request.company}\nInterview type: {request.interview_type}\n"
        f"Difficulty: {request.difficulty}\nLanguage: {request.language}\nTurn: {request.turn_number} of {request.max_turns}\n"
        f"Candidate resume context (use to tailor questions; omit contact details): {request.resume_context[:6000] or 'Not provided'}\n"
        f"Previous conversation:\n{prior}\n\nCurrent question: {request.question[:500]}\nCandidate answer: {request.answer[:4000]}"
    )
    try:
        score = max(0, min(100, int(data.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    feedback = data.get("feedback") if isinstance(data.get("feedback"), str) else "Review the answer for clarity and add a specific example."
    strengths = data.get("strengths") if isinstance(data.get("strengths"), list) else []
    improvements = data.get("improvements") if isinstance(data.get("improvements"), list) else []
    is_complete = bool(data.get("is_complete")) or request.turn_number >= request.max_turns
    next_question = data.get("next_question") if isinstance(data.get("next_question"), str) else ""
    if not is_complete and not next_question.strip():
        next_question = f"Can you give a specific example related to your work as a {request.role}?"
    return {
        "score": score,
        "feedback": feedback.strip(),
        "strengths": [str(item) for item in strengths[:4]],
        "improvements": [str(item) for item in improvements[:4]],
        "next_question": next_question.strip(),
        "is_complete": is_complete,
    }


@app.post("/interview/report")
def interview_report(request: InterviewReportRequest, user: dict[str, Any] = Depends(get_current_user)):
    history = [item.model_dump() for item in request.history]
    avg_score = round(sum(item["score"] for item in history) / len(history))
    strongest = sorted(history, key=lambda item: item["score"], reverse=True)
    needs_work = list(reversed(strongest))
    strengths = list(dict.fromkeys(
        str(point).strip()[:300]
        for item in strongest for point in item.get("strengths", []) if str(point).strip()
    ))[:5]
    if not strengths:
        strengths = [f"Stronger response ({item['score']}/100) to: {item['question'][:160]}. {item['feedback'][:220]}" for item in strongest[:3] if item["score"] >= avg_score]
    improvements = list(dict.fromkeys(
        str(point).strip()[:300]
        for item in needs_work for point in item.get("improvements", []) if str(point).strip()
    ))[:5]
    if not improvements:
        improvements = [f"Review this answer ({item['score']}/100): {item['feedback'][:220]}" for item in needs_work[:3]]
    topics = list(dict.fromkeys(item["question"].strip()[:160] for item in needs_work[:3] if item["question"].strip()))
    report = {
        "overall_score": avg_score,
        "questions_answered": len(history),
        "summary": f"Your average score was {avg_score}% across {len(history)} answers for {request.role}. This report combines the strengths and improvement points from each answer. {request.focus_changes} tab focus change(s) were recorded; this is not proof of misconduct." if request.focus_changes else f"Your average score was {avg_score}% across {len(history)} answers for {request.role}. This report combines the strengths and improvement points from each answer.",
        "strengths": strengths or ["You completed the interview and received feedback on each answer."],
        "areas_to_improve": improvements,
        "recommended_topics": topics,
        "focus_changes": request.focus_changes,
    }
    with auth_connection() as connection:
        connection.execute(
            "INSERT INTO interview_reports (id,user_id,role,company,interview_type,score,questions_answered,duration_seconds,created_at,report_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, user["id"], request.role, request.company, request.interview_type,
             report["overall_score"], report["questions_answered"], request.duration_seconds, int(time.time()),
             json.dumps(report, ensure_ascii=False)),
        )
    return report


@app.post("/resume/parse")
async def parse_resume(file: UploadFile = File(...), user: dict[str, Any] = Depends(get_current_user)):
    content = await file.read(10 * 1024 * 1024 + 1)
    return {"filename": file.filename or "resume", **parse_resume_bytes(file.filename or "", content)}


@app.post("/resume/parse-text")
def parse_resume_text(request: ResumeTextRequest, user: dict[str, Any] = Depends(get_current_user)):
    return resume_profile(request.text)


@app.get("/dashboard")
def dashboard_data(user: dict[str, Any] = Depends(get_current_user)):
    with auth_connection() as connection:
        rows = connection.execute(
            "SELECT id,role,company,interview_type,score,questions_answered,duration_seconds,created_at,report_json "
            "FROM interview_reports WHERE user_id=? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
    now = int(time.time())
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    week_start = now - (now % 86400) - (time.gmtime(now).tm_wday * 86400)
    date_set = {time.strftime("%Y-%m-%d", time.gmtime(row["created_at"])) for row in rows}
    streak = 0
    for offset in range(366):
        day = time.strftime("%Y-%m-%d", time.gmtime(now - offset * 86400))
        if day not in date_set:
            if offset == 0:
                continue
            break
        streak += 1
    recent = [{
        "id": row["id"], "role": row["role"], "company": row["company"],
        "type": row["interview_type"], "score": row["score"],
        "questions_answered": row["questions_answered"], "duration_seconds": row["duration_seconds"],
        "created_at": row["created_at"], "report": json.loads(row["report_json"]),
    } for row in rows[:8]]
    weekly_count = sum(1 for row in rows if row["created_at"] >= week_start)
    today_count = sum(1 for row in rows if time.strftime("%Y-%m-%d", time.gmtime(row["created_at"])) == today)
    return {
        "practice_sessions": len(rows),
        "average_score": round(sum(row["score"] for row in rows) / len(rows)) if rows else None,
        "practice_seconds": sum(row["duration_seconds"] for row in rows),
        "current_streak": streak,
        "weekly_sessions": weekly_count,
        "weekly_challenge_target": 5,
        "today_sessions": today_count,
        "recent_sessions": recent,
    }
