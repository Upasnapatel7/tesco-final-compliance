# """
# auth.py — Authentication System
# ================================
# Beautiful login/signup page with SQLite user store.
# Passwords hashed with SHA-256 + salt.
# Session-based authentication via Streamlit session_state.
# """

# import hashlib
# import sqlite3
# import secrets
# import os
# from pathlib import Path
# from datetime import datetime


# # ─────────────────────────────────────────────────────────────────────────────
# #  USER DATABASE
# # ─────────────────────────────────────────────────────────────────────────────
# _DB = Path("auth.db")

# def _init_db():
#     with sqlite3.connect(_DB) as c:
#         c.execute("""CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY,
#             username TEXT UNIQUE,
#             email TEXT UNIQUE,
#             password_hash TEXT,
#             salt TEXT,
#             role TEXT DEFAULT 'user',
#             created_at TEXT,
#             last_login TEXT
#         )""")
#         c.commit()

#     # ✅ Create default admin safely (NO recursion, NO errors)
#     with sqlite3.connect(_DB) as c:
#         exists = c.execute(
#             "SELECT 1 FROM users WHERE username = ?",
#             ("admin",)
#         ).fetchone()

#         if not exists:
#             salt = secrets.token_hex(16)
#             pw_hash = _hash("admin123", salt)

#             c.execute("""
#                 INSERT INTO users
#                 (username, email, password_hash, salt, role, created_at)
#                 VALUES (?,?,?,?,?,?)
#             """, (
#                 "admin",
#                 "admin@studio.ai",
#                 pw_hash,
#                 salt,
#                 "admin",
#                 datetime.now().isoformat()
#             ))
#             c.commit()
#     # Create default admin if not exists (NO recursion)
# with sqlite3.connect(str(_DB)) as c:
#     exists = c.execute(
#         "SELECT 1 FROM users WHERE username = ?",
#         ("admin",)
#     ).fetchone()

# def _count_users() -> int:
#     with sqlite3.connect(_DB) as c:
#         return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]

# def _hash(password: str, salt: str) -> str:
#     return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

# def register(username: str, email: str, password: str,
#              role: str = "user") -> tuple:
#     """Register new user. Returns (success, message)."""
#     if len(password) < 6:
#         return False, "Password must be at least 6 characters."
#     if len(username) < 3:
#         return False, "Username must be at least 3 characters."
#     salt = secrets.token_hex(16)
#     pw_hash = _hash(password, salt)
#     try:
#         with sqlite3.connect(_DB) as c:
#             c.execute("""INSERT INTO users
#                 (username, email, password_hash, salt, role, created_at)
#                 VALUES (?,?,?,?,?,?)""",
#                 (username.lower().strip(), email.lower().strip(),
#                  pw_hash, salt, role, datetime.now().isoformat()))
#             c.commit()
#         return True, "Account created successfully."
#     except sqlite3.IntegrityError as e:
#         if "username" in str(e):
#             return False, "Username already taken."
#         if "email" in str(e):
#             return False, "Email already registered."
#         return False, "Registration failed."

# def login(username_or_email: str, password: str) -> tuple:
#     """Login. Returns (success, user_dict or error_message)."""
#     _init_db()
#     with sqlite3.connect(_DB) as c:
#         row = c.execute("""SELECT id, username, email, password_hash,
#             salt, role FROM users
#             WHERE username=? OR email=?""",
#             (username_or_email.lower().strip(),
#              username_or_email.lower().strip())).fetchone()
#     if not row:
#         return False, "Username or email not found."
#     _, username, email, pw_hash, salt, role = row
#     if _hash(password, salt) != pw_hash:
#         return False, "Incorrect password."
#     # Update last login
#     with sqlite3.connect(_DB) as c:
#         c.execute("UPDATE users SET last_login=? WHERE username=?",
#                   (datetime.now().isoformat(), username))
#         c.commit()
#     return True, {"username": username, "email": email, "role": role}


# # ─────────────────────────────────────────────────────────────────────────────
# #  STREAMLIT AUTH PAGE
# # ─────────────────────────────────────────────────────────────────────────────
# def render_auth_page() -> bool:
#     """
#     Renders the login/signup page.
#     Returns True if user is authenticated (skip to main app).
#     Call at the very top of dashboard.py before any other rendering.
#     """
#     import streamlit as st

#     # Check if already logged in
#     if st.session_state.get("authenticated"):
#         return True

#     # ── Page CSS ──────────────────────────────────────────────────────────────
#     st.markdown("""
#     <style>
#     /* Hide streamlit chrome */
#     #MainMenu, footer, header { visibility: hidden; }
#     .block-container { padding: 0 !important; max-width: 100% !important; }

#     /* Auth page background */
#     .auth-bg {
#         min-height: 100vh;
#         background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0d2340 100%);
#         display: flex; align-items: center; justify-content: center;
#         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#     }
#     .auth-card {
#         background: #161b22;
#         border: 1px solid #30363d;
#         border-radius: 16px;
#         padding: 40px;
#         width: 100%;
#         max-width: 440px;
#         box-shadow: 0 24px 64px rgba(0,0,0,0.5);
#     }
#     .auth-logo {
#         font-size: 48px;
#         text-align: center;
#         margin-bottom: 8px;
#     }
#     .auth-title {
#         text-align: center;
#         font-size: 24px;
#         font-weight: 700;
#         color: #e6edf3;
#         margin-bottom: 4px;
#     }
#     .auth-subtitle {
#         text-align: center;
#         font-size: 13px;
#         color: #8b949e;
#         margin-bottom: 32px;
#     }
#     .auth-input input {
#         background: #0d1117 !important;
#         border: 1px solid #30363d !important;
#         border-radius: 8px !important;
#         color: #e6edf3 !important;
#         font-size: 15px !important;
#         padding: 12px 16px !important;
#     }
#     .auth-input input:focus {
#         border-color: #1f6feb !important;
#         box-shadow: 0 0 0 3px rgba(31,111,235,0.2) !important;
#     }
#     .stButton > button {
#         background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 8px !important;
#         padding: 12px !important;
#         font-size: 15px !important;
#         font-weight: 600 !important;
#         width: 100% !important;
#         cursor: pointer !important;
#         transition: all 0.2s !important;
#     }
#     .stButton > button:hover {
#         background: linear-gradient(135deg, #388bfd, #1f6feb) !important;
#         transform: translateY(-1px) !important;
#         box-shadow: 0 8px 24px rgba(31,111,235,0.4) !important;
#     }
#     .divider-text {
#         text-align: center;
#         color: #8b949e;
#         font-size: 12px;
#         margin: 16px 0;
#         position: relative;
#     }
#     .divider-text::before, .divider-text::after {
#         content: '';
#         position: absolute;
#         top: 50%;
#         width: 42%;
#         height: 1px;
#         background: #30363d;
#     }
#     .divider-text::before { left: 0; }
#     .divider-text::after { right: 0; }
#     .feature-pill {
#         display: inline-block;
#         background: #1f6feb22;
#         border: 1px solid #1f6feb44;
#         color: #58a6ff;
#         border-radius: 20px;
#         padding: 4px 12px;
#         font-size: 11px;
#         margin: 2px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # ── Layout: left = branding, right = form ────────────────────────────────
#     col_brand, col_form = st.columns([1.2, 1])

#     with col_brand:
#         st.markdown("""
#         <div style="padding: 60px 40px; min-height: 100vh;
#                     display: flex; flex-direction: column; justify-content: center;">

#           <div style="font-size: 52px; margin-bottom: 16px;">🎨</div>

#           <h1 style="color: #e6edf3; font-size: 32px; font-weight: 800;
#                      margin: 0 0 8px; line-height: 1.2;">
#             GenAI Creative<br>Compliance Studio
#           </h1>

#           <p style="color: #8b949e; font-size: 15px; margin: 0 0 32px;
#                     line-height: 1.6;">
#             The world's first AI system that generates advertising creatives
#             and checks compliance simultaneously — before you publish.
#           </p>

#           <div style="margin-bottom: 32px;">
#             <span class="feature-pill">🤖 65+ Brands</span>
#             <span class="feature-pill">🔍 AI Compliance</span>
#             <span class="feature-pill">🎬 Video Editor</span>
#             <span class="feature-pill">✨ AI Director</span>
#             <span class="feature-pill">📊 Analytics</span>
#             <span class="feature-pill">🧠 XAI</span>
#             <span class="feature-pill">🌍 Multi-Jurisdiction</span>
#           </div>

#           <div style="background: #0d1117; border: 1px solid #21262d;
#                       border-radius: 12px; padding: 20px;">
#             <div style="color: #3fb950; font-size: 12px; font-weight: 700;
#                         text-transform: uppercase; letter-spacing: 1px;
#                         margin-bottom: 8px;">
#               ✅ Patent-Grade AI Features
#             </div>
#             <div style="color: #c9d1d9; font-size: 13px; line-height: 1.8;">
#               · Compliance-Constrained Creative Generation<br>
#               · Multi-Modal Compliance Ensemble (MACE)<br>
#               · Brand Drift Detector (Mahalanobis distance)<br>
#               · Cross-Jurisdiction Compliance Transfer<br>
#               · Explainable AI with token-level attribution
#             </div>
#           </div>
#         </div>
#         """, unsafe_allow_html=True)

#     with col_form:
#         st.markdown("""
#         <div style="min-height: 100vh; display: flex; align-items: center;
#                     justify-content: center; padding: 40px 20px;
#                     background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);">
#         <div class="auth-card">
#         """, unsafe_allow_html=True)

#         # Tab: Login / Sign Up
#         mode = st.radio(
#     "Select Mode",
#     ["Sign In", "Create Account"],
#     label_visibility="collapsed"
# )

#         st.markdown("<br>", unsafe_allow_html=True)

#         if mode == "Sign In":
#             st.markdown('<div class="auth-logo">👋</div>', unsafe_allow_html=True)
#             st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
#             st.markdown('<div class="auth-subtitle">Sign in to your Creative Studio</div>',
#                         unsafe_allow_html=True)

#             username = st.text_input("Username or Email",
#                                       placeholder="Enter username or email",
#                                       key="login_user")
#             password = st.text_input("Password",
#                                       type="password",
#                                       placeholder="Enter password",
#                                       key="login_pass")

#             col_remember, col_forgot = st.columns(2)
#             remember = col_remember.checkbox("Remember me", key="remember")

#             if st.button("Sign In →", use_container_width=True, key="btn_login"):
#                 if not username or not password:
#                     st.error("Please enter username and password.")
#                 else:
#                     ok, result = login(username, password)
#                     if ok:
#                         st.session_state["authenticated"] = True
#                         st.session_state["user"]          = result
#                         st.success(f"Welcome back, {result['username']}! 🎨")
#                         st.rerun()
#                     else:
#                         st.error(f"❌ {result}")

#             st.markdown('<div class="divider-text">or</div>', unsafe_allow_html=True)
#             st.markdown("""
#             <div style="text-align:center; color:#8b949e; font-size:12px; margin-top:8px;">
#               Default: <code style="color:#58a6ff">admin</code> /
#               <code style="color:#58a6ff">admin123</code>
#             </div>
#             """, unsafe_allow_html=True)

#         else:
#             st.markdown('<div class="auth-logo">🚀</div>', unsafe_allow_html=True)
#             st.markdown('<div class="auth-title">Create your account</div>',
#                         unsafe_allow_html=True)
#             st.markdown('<div class="auth-subtitle">Start building compliant creatives today</div>',
#                         unsafe_allow_html=True)

#             new_username = st.text_input("Username",
#                                           placeholder="Choose a username",
#                                           key="reg_user")
#             new_email    = st.text_input("Email",
#                                           placeholder="your@email.com",
#                                           key="reg_email")
#             new_pass     = st.text_input("Password",
#                                           type="password",
#                                           placeholder="Min. 6 characters",
#                                           key="reg_pass")
#             new_pass2    = st.text_input("Confirm Password",
#                                           type="password",
#                                           placeholder="Repeat password",
#                                           key="reg_pass2")

#             if st.button("Create Account →", use_container_width=True, key="btn_reg"):
#                 if not all([new_username, new_email, new_pass, new_pass2]):
#                     st.error("Please fill in all fields.")
#                 elif new_pass != new_pass2:
#                     st.error("Passwords do not match.")
#                 else:
#                     ok, msg = register(new_username, new_email, new_pass)
#                     if ok:
#                         st.success(f"✅ {msg} Please sign in.")
#                         st.session_state["auth_mode"] = "Sign In"
#                         st.rerun()
#                     else:
#                         st.error(f"❌ {msg}")

#         st.markdown("</div></div>", unsafe_allow_html=True)

#     return False  # Not authenticated yet


# def render_user_badge():
#     """Renders user info + logout button in sidebar top."""
#     import streamlit as st
#     user = st.session_state.get("user", {})
#     username = user.get("username", "User")
#     role     = user.get("role", "user")

#     st.markdown(
#         f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;'
#         f'padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:10px">'
#         f'<div style="width:32px;height:32px;background:linear-gradient(135deg,#1f6feb,#388bfd);'
#         f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
#         f'font-size:14px;font-weight:700;color:white">{username[0].upper()}</div>'
#         f'<div><div style="color:#e6edf3;font-size:13px;font-weight:600">{username}</div>'
#         f'<div style="color:#8b949e;font-size:11px">{role.title()}</div></div>'
#         f'</div>', unsafe_allow_html=True)

#     if st.button("🚪 Sign Out", use_container_width=True, key="logout_btn"):
#         st.session_state["authenticated"] = False
#         st.session_state["user"]          = {}
#         st.rerun()

"""
auth.py -- Authentication System
=================================
Beautiful login/signup page with SQLite user store.
Passwords hashed with SHA-256 + salt.
Session-based authentication via Streamlit session_state.
"""

import hashlib
import sqlite3
import secrets
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
#  USER DATABASE
# -----------------------------------------------------------------------------
_DB = Path("auth.db")


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _init_db():
    with sqlite3.connect(_DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            salt TEXT,
            role TEXT DEFAULT 'user',
            created_at TEXT,
            last_login TEXT
        )""")
        c.commit()

    with sqlite3.connect(_DB) as c:
        exists = c.execute(
            "SELECT 1 FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if not exists:
            salt = secrets.token_hex(16)
            pw_hash = _hash("admin123", salt)
            c.execute(
                """INSERT INTO users
                   (username, email, password_hash, salt, role, created_at)
                   VALUES (?,?,?,?,?,?)""",
                ("admin", "admin@studio.ai", pw_hash, salt,
                 "admin", datetime.now().isoformat())
            )
            c.commit()


# Initialise DB on module load
_init_db()


def _count_users() -> int:
    with sqlite3.connect(_DB) as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def register(username: str, email: str, password: str,
             role: str = "user") -> tuple:
    """Register new user. Returns (success, message)."""
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    salt = secrets.token_hex(16)
    pw_hash = _hash(password, salt)
    try:
        with sqlite3.connect(_DB) as c:
            c.execute(
                """INSERT INTO users
                   (username, email, password_hash, salt, role, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (username.lower().strip(), email.lower().strip(),
                 pw_hash, salt, role, datetime.now().isoformat())
            )
            c.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already taken."
        if "email" in str(e):
            return False, "Email already registered."
        return False, "Registration failed."


def login(username_or_email: str, password: str) -> tuple:
    """Login. Returns (success, user_dict or error_message)."""
    _init_db()
    ue = username_or_email.lower().strip()
    with sqlite3.connect(_DB) as c:
        row = c.execute(
            """SELECT id, username, email, password_hash, salt, role
               FROM users WHERE username=? OR email=?""",
            (ue, ue)
        ).fetchone()
    if not row:
        return False, "Username or email not found."
    _, username, email, pw_hash, salt, role = row
    if _hash(password, salt) != pw_hash:
        return False, "Incorrect password."
    with sqlite3.connect(_DB) as c:
        c.execute("UPDATE users SET last_login=? WHERE username=?",
                  (datetime.now().isoformat(), username))
        c.commit()
    return True, {"username": username, "email": email, "role": role}


# -----------------------------------------------------------------------------
#  STREAMLIT AUTH PAGE
# -----------------------------------------------------------------------------
def render_auth_page() -> bool:
    """
    Renders the login/signup page.
    Returns True if user is authenticated.
    """
    import streamlit as st

    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; padding: 12px !important;
        font-size: 15px !important; font-weight: 600 !important;
        width: 100% !important; cursor: pointer !important;
    }
    .feature-pill {
        display: inline-block; background: #1f6feb22;
        border: 1px solid #1f6feb44; color: #58a6ff;
        border-radius: 20px; padding: 4px 12px;
        font-size: 11px; margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    col_brand, col_form = st.columns([1.2, 1])

    with col_brand:
        st.markdown("""
        <div style="padding:60px 40px;min-height:100vh;display:flex;
                    flex-direction:column;justify-content:center;">
          <div style="font-size:52px;margin-bottom:16px;">&#127912;</div>
          <h1 style="color:#e6edf3;font-size:32px;font-weight:800;
                     margin:0 0 8px;line-height:1.2;">
            GenAI Creative<br>Compliance Studio</h1>
          <p style="color:#8b949e;font-size:15px;margin:0 0 32px;line-height:1.6;">
            Generate advertising creatives and check compliance
            simultaneously -- before you publish.</p>
          <div style="margin-bottom:32px;">
            <span class="feature-pill">65+ Brands</span>
            <span class="feature-pill">AI Compliance</span>
            <span class="feature-pill">Video Editor</span>
            <span class="feature-pill">AI Director</span>
            <span class="feature-pill">Analytics</span>
            <span class="feature-pill">XAI</span>
          </div>
          <div style="background:#0d1117;border:1px solid #21262d;
                      border-radius:12px;padding:20px;">
            <div style="color:#3fb950;font-size:12px;font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;
                        margin-bottom:8px;">Patent-Grade AI Features</div>
            <div style="color:#c9d1d9;font-size:13px;line-height:1.8;">
              - Compliance-Constrained Creative Generation<br>
              - Multi-Modal Compliance Ensemble (MACE)<br>
              - Brand Drift Detector<br>
              - Cross-Jurisdiction Compliance Transfer<br>
              - Explainable AI with token-level attribution
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='padding:60px 20px;min-height:100vh;"
                    "display:flex;flex-direction:column;justify-content:center;"
                    "background:linear-gradient(180deg,#0d1117,#161b22)'>",
                    unsafe_allow_html=True)

        mode = st.radio("Mode", ["Sign In", "Create Account"],
                        label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        if mode == "Sign In":
            st.markdown("### Welcome back")
            username = st.text_input("Username or Email",
                                     placeholder="Enter username or email",
                                     key="login_user")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter password",
                                     key="login_pass")
            st.checkbox("Remember me", key="remember")

            if st.button("Sign In", use_container_width=True, key="btn_login"):
                if not username or not password:
                    st.error("Please enter username and password.")
                else:
                    ok, result = login(username, password)
                    if ok:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = result
                        st.success(f"Welcome back, {result['username']}!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {result}")

            st.markdown("""
            <div style="text-align:center;color:#8b949e;font-size:12px;margin-top:12px;">
              Default: <code style="color:#58a6ff">admin</code> /
              <code style="color:#58a6ff">admin123</code>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("### Create your account")
            new_username = st.text_input("Username", placeholder="Choose a username",
                                         key="reg_user")
            new_email    = st.text_input("Email", placeholder="your@email.com",
                                         key="reg_email")
            new_pass     = st.text_input("Password", type="password",
                                         placeholder="Min. 6 characters",
                                         key="reg_pass")
            new_pass2    = st.text_input("Confirm Password", type="password",
                                         placeholder="Repeat password",
                                         key="reg_pass2")

            if st.button("Create Account", use_container_width=True, key="btn_reg"):
                if not all([new_username, new_email, new_pass, new_pass2]):
                    st.error("Please fill in all fields.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register(new_username, new_email, new_pass)
                    if ok:
                        st.success(f"{msg} Please sign in.")
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {msg}")

        st.markdown("</div>", unsafe_allow_html=True)

    return False


def render_user_badge():
    """Renders user info + logout button in sidebar."""
    import streamlit as st
    user     = st.session_state.get("user", {})
    username = user.get("username", "User")
    role     = user.get("role", "user")

    st.markdown(
        f'<div style="background:#161b22;border:1px solid #21262d;'
        f'border-radius:8px;padding:10px 14px;margin-bottom:12px;'
        f'display:flex;align-items:center;gap:10px">'
        f'<div style="width:32px;height:32px;background:linear-gradient('
        f'135deg,#1f6feb,#388bfd);border-radius:50%;display:flex;'
        f'align-items:center;justify-content:center;font-size:14px;'
        f'font-weight:700;color:white">{username[0].upper()}</div>'
        f'<div><div style="color:#e6edf3;font-size:13px;font-weight:600">'
        f'{username}</div>'
        f'<div style="color:#8b949e;font-size:11px">{role.title()}</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    if st.button("Sign Out", use_container_width=True, key="logout_btn"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = {}
        st.rerun()