import base64
import io
import json
import os
import smtplib
import ssl
import subprocess
import tempfile
import uuid
from asyncio import sleep
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import modal
import stripe
from fasthtml import common as fh
from fasthtml.oauth import GitHubAppClient, GoogleAppClient, redir_url
from passlib.hash import pbkdf2_sha256
from PIL import Image
from simpleicons.icons import si_github
from sqlmodel import Session as DBSession
from sqlmodel import create_engine, select
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from db.models import (
    GlobalBalance,
    GlobalBalanceCreate,
    GlobalBalanceRead,
    User,
    UserRead,
    init_balance,
)
from src.utils import (
    APP_NAME,
    DEFAULT_USER_PROMPTS,
    IMAGE,
    MINUTES,
    PARENT_PATH,
    SECRETS,
)

# -----------------------------------------------------------------------------

# Modal
CPU = 8  # cores
MEM = 32768  # MB

TIMEOUT = 5 * MINUTES  # max
SCALEDOWN_WINDOW = 15 * MINUTES  # max
ALLOW_CONCURRENT_INPUTS = 1000  # max

app = modal.App(f"{APP_NAME}-frontend")

# -----------------------------------------------------------------------------


def get_app():  # noqa: C901
    # styles
    ## text
    font = "font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'"
    text_color = "stone-600"
    text_hover_color = "stone-500"  # for neutral-colored links
    text_button_hover_color = "stone-700"  # for colored buttons
    error_color = "red-400"

    ## images
    img_hover = "opacity-75"

    ## border
    border_color = "stone-400"
    border_hover_color = "stone-500"

    ## links and buttons
    click_color = "blue-400"
    click_hover_color = "blue-500"
    click_border_color = "blue-600"
    click_border_hover_color = "blue-700"

    click_neutral_color = "stone-50"
    click_neutral_hover_color = "stone-200"
    click_neutral_border_color = "stone-400"
    click_neutral_border_hover_color = "stone-500"

    click_danger_color = "red-400"
    click_danger_hover_color = "red-500"
    click_danger_border_color = "red-600"
    click_danger_border_hover_color = "red-700"

    ## effects
    rounded = "rounded-md"
    shadow = "shadow-md"
    shadow_hover = "shadow-sm"

    ## buttons
    click_button = f"{shadow} hover:{shadow_hover} text-{text_color} hover:text-{text_button_hover_color} bg-{click_color} hover:bg-{click_hover_color} border border-{click_border_color} hover:border-{click_border_hover_color}"
    click_neutral_button = f"{shadow} hover:{shadow_hover} text-{text_color} hover:text-{text_hover_color} bg-{click_neutral_color} hover:bg-{click_neutral_hover_color} border border-{click_neutral_border_color} hover:border-{click_neutral_border_hover_color}"
    click_danger_button = f"{shadow} hover:{shadow_hover} text-{text_color} hover:text-{text_button_hover_color} bg-{click_danger_color} hover:bg-{click_danger_hover_color} border border-{click_danger_border_color} hover:border-{click_danger_border_hover_color}"

    ## inputs and form pages
    input_bg_color = "stone-50"
    input_cls = f"bg-{input_bg_color} {rounded} {shadow} hover:{shadow_hover} text-{text_color} border border-{border_color} hover:border-{border_hover_color}"

    ## pages
    background_color = "stone-200"
    main_page = f"flex flex-col justify-between min-h-screen w-full bg-{background_color} text-{text_color} {font}"
    page_ctnt = "flex flex-col justify-center items-center grow gap-4 p-8"

    ## necessary hard-coded values
    tailwind_to_hex = {
        "blue-400": "#60A5FA",
        "stone-600": "#57534E",
        "stone-50": "#FAFAF9",
    }
    click_color_hex = tailwind_to_hex.get(click_color, "#60A5FA")
    text_color_hex = tailwind_to_hex.get(text_color, "#57534E")
    font_hex = (
        font.split(":")[-1].split(",")[0].strip("'").strip('"')
    )  # Extract just the font family without the CSS property part

    # setup
    def before(req, sess):
        if "session_id" not in sess:
            req.scope["session_id"] = sess.setdefault("session_id", str(uuid.uuid4()))
        if "user_uuid" not in sess:
            req.scope["user_uuid"] = sess.setdefault("user_uuid", "")

    def _not_found(req, exc):
        message = "Page not found!"
        return (
            fh.Title(APP_NAME + " | 404"),
            fh.Div(
                nav(req.session),
                fh.Main(
                    fh.P(
                        message,
                        cls=f"text-2xl text-{error_color}",
                    ),
                    cls=page_ctnt,
                ),
                toast_container(),
                cls=main_page,
            ),
        )

    f_app, _ = fh.fast_app(
        ws_hdr=True,
        before=fh.Beforeware(
            before,
            skip=[
                r"/favicon\.ico",
                r"/static/.*",
                r".*\.css",
            ],
        ),
        exception_handlers={404: _not_found},
        hdrs=[
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
            fh.Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
        ],
        boost=True,
    )
    fh.setup_toasts(f_app)
    f_app.add_middleware(
        CORSMiddleware,
        allow_origins=["/"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # db
    engine = create_engine(url=os.getenv("POSTGRES_URL"), echo=False)

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

    def get_curr_user(session) -> UserRead:
        if session["user_uuid"]:
            with get_db_session() as db_session:
                query = select(User).where(User.uuid == session["user_uuid"])
                return db_session.exec(query).first()
        return None

    def get_curr_balance() -> GlobalBalance:
        with get_db_session() as db_session:
            curr_balance = db_session.get(GlobalBalance, 1)
            if not curr_balance:
                new_balance = GlobalBalanceCreate(balance=init_balance)
                curr_balance = GlobalBalance.model_validate(new_balance)
                db_session.add(curr_balance)
                db_session.commit()
                db_session.refresh(curr_balance)
            return curr_balance

    # stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    webhook_secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    DOMAIN = os.environ["DOMAIN"]

    # SSE state
    shutdown_event = fh.signal_shutdown()
    global shown_balance
    shown_balance = 0

    # oauth
    google_client = GoogleAppClient(
        os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET")
    )
    github_client = GitHubAppClient(
        os.getenv("GITHUB_CLIENT_ID"), os.getenv("GITHUB_CLIENT_SECRET")
    )

    # layout
    def nav(session, show_auth=True):
        curr_user = get_curr_user(session)
        return fh.Nav(
            fh.Div(
                fh.A(
                    fh.Img(
                        src="/logo.png",
                        cls="w-10 h-10 object-contain",
                    ),
                    fh.P(
                        APP_NAME,
                        cls=f"text-lg text-{text_color} ",
                    ),
                    href="/",
                    cls=f"flex justify-center items-center gap-2 hover:{img_hover} hover:text-{text_hover_color}",
                ),
            ),
            fh.Svg(
                fh.NotStr(
                    """<style>
                    .spinner_zWVm { animation: spinner_5QiW 1.2s linear infinite, spinner_PnZo 1.2s linear infinite; }
                    .spinner_gfyD { animation: spinner_5QiW 1.2s linear infinite, spinner_4j7o 1.2s linear infinite; animation-delay: .1s; }
                    .spinner_T5JJ { animation: spinner_5QiW 1.2s linear infinite, spinner_fLK4 1.2s linear infinite; animation-delay: .1s; }
                    .spinner_E3Wz { animation: spinner_5QiW 1.2s linear infinite, spinner_tDji 1.2s linear infinite; animation-delay: .2s; }
                    .spinner_g2vs { animation: spinner_5QiW 1.2s linear infinite, spinner_CMiT 1.2s linear infinite; animation-delay: .2s; }
                    .spinner_ctYB { animation: spinner_5QiW 1.2s linear infinite, spinner_cHKR 1.2s linear infinite; animation-delay: .2s; }
                    .spinner_BDNj { animation: spinner_5QiW 1.2s linear infinite, spinner_Re6e 1.2s linear infinite; animation-delay: .3s; }
                    .spinner_rCw3 { animation: spinner_5QiW 1.2s linear infinite, spinner_EJmJ 1.2s linear infinite; animation-delay: .3s; }
                    .spinner_Rszm { animation: spinner_5QiW 1.2s linear infinite, spinner_YJOP 1.2s linear infinite; animation-delay: .4s; }
                    @keyframes spinner_5QiW { 0%, 50% { width: 7.33px; height: 7.33px; } 25% { width: 1.33px; height: 1.33px; } }
                    @keyframes spinner_PnZo { 0%, 50% { x: 1px; y: 1px; } 25% { x: 4px; y: 4px; } }
                    @keyframes spinner_4j7o { 0%, 50% { x: 8.33px; y: 1px; } 25% { x: 11.33px; y: 4px; } }
                    @keyframes spinner_fLK4 { 0%, 50% { x: 1px; y: 8.33px; } 25% { x: 4px; y: 11.33px; } }
                    @keyframes spinner_tDji { 0%, 50% { x: 15.66px; y: 1px; } 25% { x: 18.66px; y: 4px; } }
                    @keyframes spinner_CMiT { 0%, 50% { x: 8.33px; y: 8.33px; } 25% { x: 11.33px; y: 11.33px; } }
                    @keyframes spinner_cHKR { 0%, 50% { x: 1px; y: 15.66px; } 25% { x: 4px; y: 18.66px; } }
                    @keyframes spinner_Re6e { 0%, 50% { x: 15.66px; y: 8.33px; } 25% { x: 18.66px; y: 11.33px; } }
                    @keyframes spinner_EJmJ { 0%, 50% { x: 8.33px; y: 15.66px; } 25% { x: 11.33px; y: 18.66px; } }
                    @keyframes spinner_YJOP { 0%, 50% { x: 15.66px; y: 15.66px; } 25% { x: 18.66px; y: 18.66px; } }
                </style>
                <rect class="spinner_zWVm" x="1" y="1" width="7.33" height="7.33"/>
                <rect class="spinner_gfyD" x="8.33" y="1" width="7.33" height="7.33"/>
                <rect class="spinner_T5JJ" x="1" y="8.33" width="7.33" height="7.33"/>
                <rect class="spinner_E3Wz" x="15.66" y="1" width="7.33" height="7.33"/>
                <rect class="spinner_g2vs" x="8.33" y="8.33" width="7.33" height="7.33"/>
                <rect class="spinner_ctYB" x="1" y="15.66" width="7.33" height="7.33"/>
                <rect class="spinner_BDNj" x="15.66" y="8.33" width="7.33" height="7.33"/>
                <rect class="spinner_rCw3" x="8.33" y="15.66" width="7.33" height="7.33"/>
                <rect class="spinner_Rszm" x="15.66" y="15.66" width="7.33" height="7.33"/>
                """
                ),
                id="spinner",
                cls=f"htmx-indicator w-10 h-10 object-contain absolute top-6 left-1/2 transform -translate-x-1/2 fill-{click_color}",
            ),
            fh.Div(
                fh.A(
                    fh.P(
                        "Trials",
                        cls=f"text-{text_color} hover:text-{text_hover_color}",
                    ),
                    href="/trials",
                ),
                overlay_close(session),
                cls="flex flex-col md:flex-row justify-center items-end md:items-center gap-4 md:gap-6",
            )
            if curr_user and show_auth
            else fh.Div(
                fh.A(
                    fh.P(
                        "Log In",
                        cls=f"text-{text_color} hover:text-{text_hover_color}",
                    ),
                    href="/login",
                ),
                fh.A(
                    fh.Button(
                        "Sign Up",
                        cls=f"{click_button} {rounded} px-4 py-2",
                    ),
                    href="/signup",
                ),
                cls="flex flex-col md:flex-row justify-center items-end md:items-center gap-4 md:gap-6",
            )
            if show_auth
            else None,
            cls="relative flex justify-between items-center p-4",
        )

    def home_content():
        return fh.Main(
            fh.Form(
                fh.Textarea(
                    id="user-prompt",
                    name="user_prompt",
                    placeholder="Ask a research question...",
                    rows=10,
                    cls=f"{input_cls} p-4 resize-none",
                    autofocus=True,
                    onkeydown="if(event.key === 'Enter' && event.shiftKey) { this.form.submit(); event.preventDefault(); }",
                ),
                fh.Button(
                    fh.P(
                        "→",
                        cls=f"text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    type="submit",
                    cls=f"absolute bottom-0 right-4 rounded-full p-2 {click_button}",
                    style="width: 3rem; height: 3rem;",
                ),
                hx_post="/",
                hx_indicator="#spinner",
                cls="w-full md:w-2/3 relative flex justify-between items-center",
            ),
            fh.Div(
                fh.P(
                    "Or try some examples:",
                    cls=f"text-{text_color}",
                ),
                cls="w-full md:w-2/3 flex justify-start items-center pl-2",
            ),
            fh.Div(
                *[
                    fh.Button(
                        prompt,
                        cls=f"{click_button} px-4 py-2 {rounded}",
                        hx_post="/fill-user-prompt",
                        hx_indicator="#spinner",
                        hx_target="#user-prompt",
                        hx_swap="innerHTML",
                        hx_vals=json.dumps(
                            {
                                "prompt": prompt,
                            }
                        ),
                    )
                    for prompt in DEFAULT_USER_PROMPTS
                ],
                cls="w-full md:w-2/3 flex flex-col md:flex-row justify-start items-start md:items-center gap-4",
            ),
            cls=page_ctnt,
        )

    def signup_content(req, session):
        return fh.Main(
            fh.Div(
                fh.H1(
                    "Sign Up",
                    cls=f"text-4xl font-bold text-{text_color} text-center",
                ),
                fh.P(
                    fh.Span("(Already have an account? "),
                    fh.A(
                        "Log In",
                        href="/login",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    fh.Span(")"),
                    cls=f"text-{text_color} text-center",
                ),
                fh.A(
                    fh.Button(
                        fh.Div(
                            fh.Svg(
                                fh.NotStr(
                                    si_github.svg,
                                ),
                                cls="w-6 h-6",
                            ),
                            fh.Div(
                                "Continue with GitHub",
                                cls="flex grow justify-center items-center",
                            ),
                            cls="flex justify-between items-center",
                        ),
                        cls=f"w-full {click_neutral_button} p-3 {rounded}",
                    ),
                    href=github_client.login_link(redir_url(req, "/redirect-github")),
                    cls="w-full",
                ),
                fh.A(
                    fh.Button(
                        fh.Div(
                            fh.Svg(
                                fh.NotStr(
                                    """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12.0003 4.75C13.7703 4.75 15.3553 5.36002 16.6053 6.54998L20.0303 3.125C17.9502 1.19 15.2353 0 12.0003 0C7.31028 0 3.25527 2.69 1.28027 6.60998L5.27028 9.70498C6.21525 6.86002 8.87028 4.75 12.0003 4.75Z" fill="#EA4335"/><path d="M23.49 12.275C23.49 11.49 23.415 10.73 23.3 10H12V14.51H18.47C18.18 15.99 17.34 17.25 16.08 18.1L19.945 21.1C22.2 19.01 23.49 15.92 23.49 12.275Z" fill="#4285F4"/><path d="M5.26498 14.2949C5.02498 13.5699 4.88501 12.7999 4.88501 11.9999C4.88501 11.1999 5.01998 10.4299 5.26498 9.7049L1.275 6.60986C0.46 8.22986 0 10.0599 0 11.9999C0 13.9399 0.46 15.7699 1.28 17.3899L5.26498 14.2949Z" fill="#FBBC05"/><path d="M12.0004 24.0001C15.2404 24.0001 17.9654 22.935 19.9454 21.095L16.0804 18.095C15.0054 18.82 13.6204 19.245 12.0004 19.245C8.8704 19.245 6.21537 17.135 5.2654 14.29L1.27539 17.385C3.25539 21.31 7.3104 24.0001 12.0004 24.0001Z" fill="#34A853"/></svg>"""
                                ),
                                cls="w-6 h-6",
                            ),
                            fh.Div(
                                "Continue with Google",
                                cls="flex grow justify-center items-center",
                            ),
                            cls="flex justify-between items-center",
                        ),
                        cls=f"w-full {click_neutral_button} p-3 {rounded}",
                    ),
                    href=google_client.login_link(redir_url(req, "/redirect-google")),
                    cls="w-full",
                ),
                fh.Div(
                    fh.Div(cls=f"flex-grow border-t border-{text_color}"),
                    fh.Span("or", cls=f"flex-shrink px-4 text-{text_color}"),
                    fh.Div(cls=f"flex-grow border-t border-{text_color}"),
                    cls="w-full relative flex justify-center items-center",
                ),
                fh.Form(
                    fh.Input(
                        id="email",
                        name="email",
                        type="email",
                        placeholder="Email",
                        required=True,
                        cls=f"w-full {input_cls}",
                    ),
                    fh.Input(
                        id="password",
                        name="password",
                        type="password",
                        placeholder="Password",
                        required=True,
                        cls=f"w-full {input_cls}",
                    ),
                    fh.Button(
                        "Sign Up",
                        type="submit",
                        cls=f"w-full {click_button} p-3 {rounded}",
                    ),
                    hx_post="/signup",
                    hx_indicator="#spinner",
                    hx_target="this",
                    hx_swap="none",
                    onkeydown="if(event.key === 'Enter') { this.form.submit(); event.preventDefault(); }",
                    cls="w-full",
                ),
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=page_ctnt,
        )

    def login_content(req, session):
        return fh.Main(
            fh.Div(
                fh.H1(
                    "Log In",
                    cls=f"text-4xl font-bold text-{text_color} text-center",
                ),
                fh.P(
                    fh.Span("(No account? "),
                    fh.A(
                        "Sign Up",
                        href="/signup",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    fh.Span(")"),
                    cls=f"text-{text_color} text-center",
                ),
                fh.A(
                    fh.Button(
                        fh.Div(
                            fh.Svg(
                                fh.NotStr(
                                    si_github.svg,
                                ),
                                cls="w-6 h-6",
                            ),
                            fh.Div(
                                "Continue with GitHub",
                                cls="flex grow justify-center items-center",
                            ),
                            cls="flex justify-between items-center",
                        ),
                        cls=f"w-full {click_neutral_button} p-3 {rounded}",
                    ),
                    href=github_client.login_link(redir_url(req, "/redirect-github")),
                    cls="w-full",
                ),
                fh.A(
                    fh.Button(
                        fh.Div(
                            fh.Svg(
                                fh.NotStr(
                                    """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12.0003 4.75C13.7703 4.75 15.3553 5.36002 16.6053 6.54998L20.0303 3.125C17.9502 1.19 15.2353 0 12.0003 0C7.31028 0 3.25527 2.69 1.28027 6.60998L5.27028 9.70498C6.21525 6.86002 8.87028 4.75 12.0003 4.75Z" fill="#EA4335"/><path d="M23.49 12.275C23.49 11.49 23.415 10.73 23.3 10H12V14.51H18.47C18.18 15.99 17.34 17.25 16.08 18.1L19.945 21.1C22.2 19.01 23.49 15.92 23.49 12.275Z" fill="#4285F4"/><path d="M5.26498 14.2949C5.02498 13.5699 4.88501 12.7999 4.88501 11.9999C4.88501 11.1999 5.01998 10.4299 5.26498 9.7049L1.275 6.60986C0.46 8.22986 0 10.0599 0 11.9999C0 13.9399 0.46 15.7699 1.28 17.3899L5.26498 14.2949Z" fill="#FBBC05"/><path d="M12.0004 24.0001C15.2404 24.0001 17.9654 22.935 19.9454 21.095L16.0804 18.095C15.0054 18.82 13.6204 19.245 12.0004 19.245C8.8704 19.245 6.21537 17.135 5.2654 14.29L1.27539 17.385C3.25539 21.31 7.3104 24.0001 12.0004 24.0001Z" fill="#34A853"/></svg>"""
                                ),
                                cls="w-6 h-6",
                            ),
                            fh.Div(
                                "Continue with Google",
                                cls="flex grow justify-center items-center",
                            ),
                            cls="flex justify-between items-center",
                        ),
                        cls=f"w-full {click_neutral_button} p-3 {rounded}",
                    ),
                    href=google_client.login_link(redir_url(req, "/redirect-google")),
                    cls="w-full",
                ),
                fh.Div(
                    fh.Div(cls=f"flex-grow border-t border-{text_color}"),
                    fh.Span("or", cls=f"flex-shrink mx-4 text-{text_color}"),
                    fh.Div(cls=f"flex-grow border-t border-{text_color}"),
                    cls="w-full relative flex justify-center items-center",
                ),
                fh.Form(
                    fh.Input(
                        id="email",
                        name="email",
                        type="email",
                        placeholder="Email",
                        required=True,
                        cls=f"w-full {input_cls}",
                    ),
                    fh.Input(
                        id="password",
                        name="password",
                        type="password",
                        placeholder="Password",
                        required=True,
                        cls=f"w-full {input_cls}",
                    ),
                    fh.Button(
                        "Log In",
                        type="submit",
                        cls=f"w-full {click_button} p-3 {rounded}",
                    ),
                    hx_post="/login",
                    hx_indicator="#spinner",
                    hx_target="this",
                    hx_swap="none",
                    onkeydown="if(event.key === 'Enter') { this.form.submit(); event.preventDefault(); }",
                    cls="w-full",
                ),
                fh.Div(
                    fh.A(
                        "Forgot Password?",
                        href="/forgot-password",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    cls="w-full",
                ),
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=page_ctnt,
        )

    def forgot_password_content():
        return fh.Main(
            fh.Form(
                fh.Div(
                    fh.A(
                        "←",
                        href="/login",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    fh.H1("Forgot Password?", cls=f"text-{text_color}"),
                    cls="w-full flex justify-center items-center gap-2 text-4xl font-bold",
                ),
                fh.P(
                    "Enter your email below to receive a password reset link.",
                    cls=f"text-{text_color} text-center",
                ),
                fh.Input(
                    type="email",
                    name="email",
                    placeholder="Email",
                    cls=f"w-full {input_cls}",
                ),
                fh.Button(
                    "Send Reset Link",
                    type="submit",
                    cls=f"w-full {click_button} p-3 {rounded}",
                ),
                hx_post="/forgot-password",
                hx_target="this",
                hx_swap="none",
                hx_indicator="#spinner",
                onkeydown="if(event.key === 'Enter') { this.form.submit(); event.preventDefault(); }",
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=page_ctnt,
        )

    def reset_password_content(token: str | None = None):
        return fh.Main(
            fh.Form(
                fh.Div(
                    fh.A(
                        "←",
                        href="/login",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    fh.H1(
                        "Reset Password",
                        cls=f"text-{text_color}",
                    ),
                    cls="w-full flex justify-center items-center gap-2 text-4xl font-bold",
                ),
                fh.P(
                    "Enter your new password below.",
                    cls=f"text-{text_color} text-center",
                ),
                fh.Input(
                    name="password",
                    placeholder="New Password",
                    type="password",
                    required=True,
                    cls=f"w-full {input_cls}",
                ),
                fh.Input(
                    name="confirm_password",
                    placeholder="Confirm Password",
                    type="password",
                    required=True,
                    cls=f"w-full {input_cls}",
                ),
                fh.Input(
                    type="hidden",
                    name="token",
                    value=token,
                )
                if token
                else "",
                fh.Button(
                    "Reset Password",
                    type="submit",
                    cls=f"w-full {click_button} p-3 {rounded}",
                ),
                hx_post="/reset-password",
                hx_target="this",
                hx_swap="none",
                hx_indicator="#spinner",
                onkeydown="if(event.key === 'Enter') { this.form.submit(); event.preventDefault(); }",
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=page_ctnt,
        )

    def trials_content():
        return fh.Main(
            fh.H1("Trials", cls=f"text-4xl font-bold text-{text_color} text-center"),
            cls=page_ctnt,
        )

    def settings_content(session):
        curr_user = get_curr_user(session)
        max_text_length_sm = 7
        max_text_length_md = 17
        return fh.Main(
            fh.Div(
                fh.H1(
                    "Settings",
                    cls=f"text-4xl font-bold text-{text_color} text-center",
                ),
                fh.Div(
                    fh.Div(
                        fh.Div(
                            fh.Img(
                                src=curr_user.profile_img,
                                cls="w-12 h-12 object-cover rounded-full",
                            ),
                            fh.Button(
                                "Edit",
                                hx_get="/settings/edit",
                                hx_indicator="#spinner",
                                hx_target="#settings",
                                hx_swap="outerHTML",
                                cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {click_button} {rounded} p-3",
                            ),
                            cls="w-full flex justify-between items-center",
                        ),
                        fh.Div(
                            fh.P("Email:", cls=f"text-{text_color}"),
                            fh.P(
                                curr_user.email
                                if len(curr_user.email) <= max_text_length_md
                                else curr_user.email[:max_text_length_md] + "...",
                                cls=f"text-{text_color} hidden md:block",
                            ),
                            fh.P(
                                curr_user.email
                                if len(curr_user.email) <= max_text_length_sm
                                else curr_user.email[:max_text_length_sm] + "...",
                                cls=f"text-{text_color} block md:hidden",
                            ),
                            cls="w-full flex justify-between items-center",
                        ),
                        fh.Div(
                            fh.P("Username:", cls=f"text-{text_color}"),
                            fh.P(
                                curr_user.username
                                if len(curr_user.username) <= max_text_length_md
                                else curr_user.username[:max_text_length_md] + "...",
                                cls=f"text-{text_color} hidden md:block",
                            ),
                            fh.P(
                                curr_user.username
                                if len(curr_user.username) <= max_text_length_sm
                                else curr_user.username[:max_text_length_sm] + "...",
                                cls=f"text-{text_color} block md:hidden",
                            ),
                            cls="w-full flex justify-between items-center",
                        ),
                        fh.Div(
                            fh.P("Password:", cls=f"text-{text_color}"),
                            fh.P("********", cls=f"text-{text_color}"),
                            cls="w-full flex justify-between items-center",
                        )
                        if curr_user.login_type == "email"
                        else None,
                        id="settings",
                        cls="w-full flex flex-col justify-between items-center gap-4",
                    ),
                    fh.Button(
                        "Delete Account",
                        hx_delete="/settings/delete-account",
                        hx_confirm="Are you sure you want to delete your account? This action cannot be undone.",
                        hx_indicator="#spinner",
                        cls=f"w-full flex justify-center items-center {click_danger_button} {rounded} p-3",
                    ),
                    cls="w-full flex flex-col justify-center items-center gap-4",
                ),
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-8 {input_cls} p-8",
            ),
            cls=page_ctnt,
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
            fh.Div(
                fh.Div(
                    fh.P("Global balance:", cls=f"text-{text_color}"),
                    fh.P(
                        f"{GlobalBalanceRead.model_validate(get_curr_balance()).balance} credits",
                        cls=f"text-{text_color} font-bold",
                        hx_ext="sse",
                        sse_connect="/stream-balance",
                        sse_swap="UpdateBalance",
                    ),
                    cls="flex justify-center items-start gap-0.5 md:gap-1",
                ),
                fh.P(
                    fh.A(
                        "Buy 5 more",
                        href="/buy_global",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    " to share ($1)",
                    cls=f"text-{text_color}",
                ),
                cls="flex flex-col justify-center items-start gap-0.5",
            ),
            fh.Div(
                fh.A(
                    fh.Svg(
                        fh.NotStr(
                            si_github.svg,
                        ),
                        cls=f"w-10 h-10 object-contain hover:{img_hover}",
                    ),
                    href="https://github.com/andrewhinh/sim",
                    target="_blank",
                ),
                fh.Div(
                    fh.P("Made by", cls=f"text-{text_color}"),
                    fh.A(
                        "Andrew Hinh",
                        href="https://ajhinh.com/",
                        cls=f"text-{click_color} hover:text-{click_hover_color}",
                    ),
                    cls="flex flex-col justify-center items-end gap-0.5",
                ),
                cls="flex flex-col md:flex-row justify-center items-end md:items-center gap-2 md:gap-4",
            ),
            cls="flex justify-between items-end md:items-center p-4 text-sm md:text-lg",
        )

    # helper fns
    async def stream_balance_updates():
        while not shutdown_event.is_set():
            curr_balance = get_curr_balance().balance
            global shown_balance
            if shown_balance != curr_balance:
                shown_balance = curr_balance
                yield f"""event: UpdateBalance\ndata: {fh.to_xml(fh.P(f"{shown_balance} credits", cls=f"text-{text_color} font-bold", sse_swap="UpdateBalance"))}\n\n"""
            await sleep(1)

    def validate_image_base64(image_base64: str) -> dict[str, str]:
        max_file_size = 10  # MB
        max_img_dim = (4096, 4096)

        # Verify MIME type and magic #
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        try:
            img.verify()
        except Exception as e:
            return {"error": e}

        # Limit img size
        if len(image_base64) > max_file_size * 1024 * 1024:
            return {"error": f"File size exceeds {max_file_size}MB limit."}
        if img.size[0] > max_img_dim[0] or img.size[1] > max_img_dim[1]:
            return {
                "error": f"Image dimensions exceed {max_img_dim[0]}x{max_img_dim[1]} pixels limit."
            }

        # Run antivirus
        # write image_base64 to tmp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(base64.b64decode(image_base64))
            tmp_file_path = tmp_file.name

        try:
            result = subprocess.run(  # noqa: S603
                ["python", "main.py", str(tmp_file_path)],  # noqa: S607
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PARENT_PATH / "Python-Antivirus",
            )
            scan_result = result.stdout.strip().lower()
            if scan_result == "infected":
                return {"error": "Potential threat detected."}
        except Exception as e:
            return {"error": f"Error during antivirus scan: {e}"}

        return {"success": image_base64}

    def validate_image_file(
        image_file,
    ) -> dict[str, str]:
        if image_file is not None:
            valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
            file_extension = Path(image_file.filename).suffix.lower()
            if file_extension not in valid_extensions:
                return {"error": "Invalid file type. Please upload an image."}
            image_file.file.seek(0)  # reset pointer in case of multiple uploads
            img_bytes = image_file.file.read()
            image_base64 = base64.b64encode(img_bytes).decode("utf-8")
            return validate_image_base64(image_base64)
        return {"error": "No image uploaded"}

    def send_password_reset_email(email, reset_link):
        token_expiry = 24  # hours

        # Email configuration
        sender_email = os.environ.get("EMAIL_SENDER", "noreply@simproject.com")
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_username = os.environ.get("SMTP_USERNAME", "")
        smtp_password = os.environ.get("SMTP_PASSWORD", "")

        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Password Reset Request"
        message["From"] = sender_email
        message["To"] = email

        # Create plain text version of email
        text = f"""
        Password Reset Request

        You requested a password reset for your account. Please click the link below to reset your password:

        {reset_link}

        If you did not request this password reset, please ignore this email.

        This link will expire in {token_expiry} hours.
        """

        # Create HTML version of email
        html = f"""
        <html>
          <head></head>
          <body>
            <h2>Password Reset Request</h2>
            <p>You requested a password reset for your account. Please click the link below to reset your password:</p>
            <p><a href="{reset_link}">Reset Password</a></p>
            <p>If you did not request this password reset, please ignore this email.</p>
            <p>This link will expire in {token_expiry} hours.</p>
          </body>
        </html>
        """

        # Attach both versions to the message
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        try:
            # Create secure connection with server and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                if smtp_username and smtp_password:
                    server.login(smtp_username, smtp_password)
                server.sendmail(sender_email, email, message.as_string())
                return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False

    # pages
    @f_app.get("/")
    def home(
        session,
    ):
        return (
            fh.Title(APP_NAME),
            fh.Div(
                nav(session),
                home_content(),
                toast_container(),
                footer(),
                cls=main_page,
            ),
        )

    @f_app.get("/signup")
    def signup_page(req, session):
        if session.get("user_uuid"):
            return fh.RedirectResponse("/")
        return (
            fh.Title(f"{APP_NAME} | sign up"),
            fh.Div(
                nav(session, False),
                signup_content(req, session),
                toast_container(),
                cls=main_page,
            ),
        )

    @f_app.get("/login")
    def login_page(req, session):
        if session.get("user_uuid"):
            return fh.RedirectResponse("/")
        return (
            fh.Title(f"{APP_NAME} | log in"),
            fh.Div(
                nav(session, False),
                login_content(req, session),
                toast_container(),
                cls=main_page,
            ),
        )

    @f_app.get("/forgot-password")
    def forgot_password_page(session):
        if session.get("user_uuid"):
            return fh.RedirectResponse("/login")
        return (
            fh.Title(f"{APP_NAME} | forgot password"),
            fh.Div(
                nav(session, False),
                forgot_password_content(),
                toast_container(),
                cls=main_page,
            ),
        )

    @f_app.get("/reset-password")
    def reset_password_page(req, session):
        if session.get("user_uuid"):
            return fh.RedirectResponse("/login")
        token = req.query_params.get("token")
        return (
            fh.Title(f"{APP_NAME} | reset password"),
            fh.Div(
                nav(session, False),
                reset_password_content(token),
                toast_container(),
                cls=main_page,
            ),
        )

    @f_app.get("/trials")
    def trials_page(session):
        if not session.get("user_uuid"):
            return fh.RedirectResponse("/login")
        return (
            fh.Title(f"{APP_NAME} | trials"),
            fh.Div(
                nav(session, True),
                trials_content(),
                toast_container(),
                cls=main_page,
            ),
        )

    @f_app.get("/settings")
    def settings_page(session):
        if not session.get("user_uuid"):
            return fh.RedirectResponse("/login")
        return (
            fh.Title(f"{APP_NAME} | settings"),
            fh.Div(
                nav(session, True),
                settings_content(session),
                toast_container(),
                cls=main_page,
            ),
        )

    # routes
    ## main page
    ### overlay
    @f_app.get("/user/overlay/show")
    def overlay_show(session):
        curr_user = get_curr_user(session)
        max_username_length = 17
        return fh.Div(
            fh.Div(
                fh.Img(
                    src=curr_user.profile_img,
                    cls=f"w-10 h-10 object-cover hover:{img_hover} rounded-full",
                ),
                fh.P(
                    curr_user.username
                    if len(curr_user.username) <= max_username_length
                    else curr_user.username[:max_username_length] + "...",
                    cls=f"text-{text_color} hover:text-{text_hover_color} hidden md:block",
                ),
                hx_get="/user/overlay/close",
                hx_trigger="click",
                hx_target="#overlay",
                hx_swap="outerHTML",
                hx_indicator="#spinner",
                cls=f"flex justify-center items-center gap-2 cursor-pointer hover:{img_hover} hover:text-{text_hover_color}",
            ),
            fh.Div(
                fh.A(
                    fh.Button(
                        "Settings",
                        cls=f"w-full {click_neutral_button} p-3 {rounded}",
                    ),
                    href="/settings",
                    cls="w-full",
                ),
                fh.Button(
                    "Log out",
                    cls=f"w-full {click_button} {rounded} p-3",
                    hx_get="/logout",
                    hx_indicator="#spinner",
                ),
                cls=f"absolute top-14 right-0 min-w-40 z-10 flex flex-col justify-center items-center gap-2 {input_cls} p-2",
            ),
            id="overlay",
            hx_swap_oob="true",
            cls="relative",
        )

    @f_app.get("/user/overlay/close")
    def overlay_close(session):
        curr_user = get_curr_user(session)
        max_username_length = 17
        return fh.Div(
            fh.Img(
                src=curr_user.profile_img,
                cls=f"w-10 h-10 object-cover hover:{img_hover} rounded-full",
            ),
            fh.P(
                curr_user.username
                if len(curr_user.username) <= max_username_length
                else curr_user.username[:max_username_length] + "...",
                cls=f"text-{text_color} hover:text-{text_hover_color} hidden md:block",
            ),
            id="overlay",
            hx_swap_oob="true",
            hx_get="/user/overlay/show",
            hx_trigger="click",
            hx_target="#overlay",
            hx_swap="outerHTML",
            hx_indicator="#spinner",
            cls=f"flex justify-center items-center gap-2 cursor-pointer hover:{img_hover} hover:text-{text_hover_color}",
        )

    ### balance sse
    @f_app.get("/stream-balance")
    async def stream_balance():
        """Stream balance updates to connected clients"""
        return StreamingResponse(
            stream_balance_updates(), media_type="text/event-stream"
        )

    ### fill user prompt
    @f_app.post("/fill-user-prompt")
    def fill_user_prompt(session, prompt: str):
        return prompt

    ### sim trial
    # @f_app.post("/")
    # def sim_trial(session, user_prompt: str):
    #     pass

    ## auth
    @f_app.get("/logout")
    def logout(session):
        if session.get("user_uuid"):
            del session["user_uuid"]
        return fh.Redirect("/")

    ### login
    @f_app.post("/login")
    def email_login(session, email: str, password: str):
        if not email:
            fh.add_toast(session, "Email is required", "error")
            return None
        if not password:
            fh.add_toast(session, "Password is required", "error")
            return None
        with get_db_session() as db_session:
            query = select(User).where(User.email == email)
            db_user = db_session.exec(query).first()
            if not db_user:
                fh.add_toast(session, "User not found", "error")
                return fh.Redirect("/signup")
            else:
                if not pbkdf2_sha256.verify(password, db_user.hashed_password):
                    fh.add_toast(session, "Incorrect credentials", "error")
                    return None
                else:
                    session["user_uuid"] = db_user.uuid
                    return fh.Redirect("/")

    ### signup
    @f_app.post("/signup")
    def email_signup(session, email: str, password: str):
        if not email:
            fh.add_toast(session, "Email is required", "error")
            return None
        if not password:
            fh.add_toast(session, "Password is required", "error")
            return None
        with get_db_session() as db_session:
            query = select(User).where(User.email == email)
            db_user = db_session.exec(query).first()
            if db_user:
                fh.add_toast(session, "User already exists", "error")
                return fh.Redirect("/login")
            else:
                extra_data = {
                    "hashed_password": pbkdf2_sha256.hash(password),
                }
                db_user = User.model_validate(
                    {
                        "login_type": "email",
                        "profile_img": f"data:image/svg+xml;base64,{base64.b64encode(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"50\" fill=\"{click_color_hex}\"/><text x=\"50\" y=\"60\" font-size=\"40\" text-anchor=\"middle\" fill=\"{text_color_hex}\" font-family=\"{font_hex}\">{email[0].upper()}</text></svg>'.encode()).decode()}",
                        "email": email,
                        "username": email,
                    },
                    update=extra_data,
                )
                db_session.add(db_user)
                db_session.commit()
                db_session.refresh(db_user)
                session["user_uuid"] = db_user.uuid
                return fh.Redirect("/")

    ## password reset
    @f_app.post("/forgot-password")
    def forgot_password(session, email: str):
        if not email:
            fh.add_toast(session, "Email is required", "error")
            return None

        token_expiry = 24  # hours
        success_msg = (
            "If an account with that email exists, a password reset link has been sent."
        )

        with get_db_session() as db_session:
            query = select(User).where(User.email == email)
            db_user = db_session.exec(query).first()
            if not db_user:
                fh.add_toast(
                    session,
                    success_msg,
                    "success",
                )
                return fh.Redirect("/login")
            if db_user.login_type != "email":
                fh.add_toast(
                    session, "This account uses a different login method.", "error"
                )
                return None

            reset_token = str(uuid.uuid4())
            with get_db_session() as db_session:
                db_user.reset_token = reset_token
                db_user.reset_token_expiry = datetime.now() + timedelta(
                    hours=token_expiry
                )
                db_session.add(db_user)
                db_session.commit()
                db_session.refresh(db_user)
            reset_link = f"{os.environ['DOMAIN']}/reset-password?token={reset_token}"
            send_password_reset_email(email, reset_link)

            fh.add_toast(
                session,
                success_msg,
                "success",
            )
            return fh.Redirect("/login")

    @f_app.post("/reset-password")
    def reset_password(session, password: str, confirm_password: str, token: str):
        if not password:
            fh.add_toast(session, "Password is required", "error")
            return None
        if not confirm_password:
            fh.add_toast(session, "Confirm password is required", "error")
            return None
        if password != confirm_password:
            fh.add_toast(session, "Passwords do not match", "error")
            return None
        if not token:
            fh.add_toast(session, "Invalid reset token", "error")
            return fh.Redirect("/login")

        with get_db_session() as db_session:
            query = select(User).where(User.reset_token == token)
            db_user = db_session.exec(query).first()
            if not db_user:
                fh.add_toast(session, "Invalid reset token", "error")
                return fh.Redirect("/login")
            if db_user.reset_token_expiry < datetime.now():
                fh.add_toast(session, "Reset token has expired", "error")
                return fh.Redirect("/login")

            db_user.hashed_password = pbkdf2_sha256.hash(password)
            db_user.reset_token = None
            db_user.reset_token_expiry = None
            db_session.add(db_user)
            db_session.commit()
            db_session.refresh(db_user)
            fh.add_toast(session, "Password has been reset", "success")
            return fh.Redirect("/login")

    ## redirect
    @f_app.get("/redirect-github")
    def redirect_github(code: str, request, session):
        if not code:
            fh.add_toast(session, "Invalid code", "error")
            return fh.Redirect("/login")

        redir = redir_url(request, "/redirect-github")
        user_info = github_client.retr_info(code, redir)
        email = user_info.get("email", "")
        username = user_info.get("login", "")

        with get_db_session() as db_session:
            query = select(User).where(User.email == email)
            db_user = db_session.exec(query).first()
            if not db_user:
                db_user = User.model_validate(
                    {
                        "login_type": "github",
                        "profile_img": f"data:image/svg+xml;base64,{base64.b64encode(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"50\" fill=\"{click_color_hex}\"/><text x=\"50\" y=\"60\" font-size=\"40\" text-anchor=\"middle\" fill=\"{text_color_hex}\" font-family=\"{font_hex}\">{email[0].upper() if email else username[0].upper()}</text></svg>'.encode()).decode()}",
                        "email": email,
                        "username": username,
                    }
                )
                db_session.add(db_user)
                db_session.commit()
                db_session.refresh(db_user)

            session["user_uuid"] = db_user.uuid
            return fh.RedirectResponse("/", status_code=303)

    @f_app.get("/redirect-google")
    def redirect_google(code: str, request, session):
        if not code:
            fh.add_toast(session, "Invalid code", "error")
            return fh.Redirect("/login")

        redir = redir_url(request, "/redirect-google")
        user_info = google_client.retr_info(code, redir)
        email = user_info.get("email", "")
        username = email.split("@")[0] if email else ""

        with get_db_session() as db_session:
            query = select(User).where(User.email == email)
            db_user = db_session.exec(query).first()
            if not db_user:
                db_user = User.model_validate(
                    {
                        "login_type": "google",
                        "profile_img": f"data:image/svg+xml;base64,{base64.b64encode(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"50\" fill=\"{click_color_hex}\"/><text x=\"50\" y=\"60\" font-size=\"40\" text-anchor=\"middle\" fill=\"{text_color_hex}\" font-family=\"{font_hex}\">{email[0].upper() if email else "U"}</text></svg>'.encode()).decode()}",
                        "email": email,
                        "username": username,
                    }
                )
                db_session.add(db_user)
                db_session.commit()
                db_session.refresh(db_user)

            session["user_uuid"] = db_user.uuid
            return fh.RedirectResponse("/", status_code=303)

    ## settings
    @f_app.get("/settings/edit")
    def edit_settings(session):
        curr_user = get_curr_user(session)
        return (
            fh.Form(
                fh.Div(
                    fh.Label(
                        fh.Input(
                            id="new-profile-img-upload",
                            name="profile_img_file",
                            type="file",
                            accept="image/*",
                            hx_post="/update-preview",
                            hx_trigger="change delay:200ms changed",
                            hx_target="#profile-image-preview",
                            hx_swap="outerHTML",
                            hx_indicator="#spinner",
                            hx_encoding="multipart/form-data",
                            cls="hidden",
                        ),
                        fh.Img(
                            src=curr_user.profile_img,
                            cls=f"w-12 h-12 object-cover rounded-full cursor-pointer hover:{img_hover}",
                            id="profile-image-preview",
                        ),
                    ),
                    fh.Button(
                        "Save",
                        hx_patch="/settings/save",
                        hx_indicator="#spinner",
                        hx_target="#settings",
                        hx_swap="outerHTML",
                        hx_include="#new-profile-img-upload",
                        hx_encoding="multipart/form-data",
                        cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {click_button} {rounded} p-3",
                    ),
                    cls="w-full flex justify-between items-center",
                ),
                fh.Div(
                    fh.P("Email:", cls=f"text-{text_color}"),
                    fh.Input(
                        value=curr_user.email,
                        name="email",
                        type="email",
                        cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {input_cls}",
                    ),
                    cls="w-full flex justify-between items-center",
                ),
                fh.Div(
                    fh.P("Username:", cls=f"text-{text_color}"),
                    fh.Input(
                        value=curr_user.username,
                        name="username",
                        cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {input_cls}",
                    ),
                    cls="w-full flex justify-between items-center",
                ),
                fh.Div(
                    fh.P("Password:", cls=f"text-{text_color}"),
                    fh.Input(
                        value="",
                        name="password",
                        type="password",
                        cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {input_cls}",
                    ),
                    cls="w-full flex justify-between items-center",
                )
                if curr_user.login_type == "email"
                else None,
                id="settings",
                cls="w-full flex flex-col justify-between items-center gap-4",
            ),
        )

    @f_app.post("/update-preview")
    def update_preview(
        session,
        profile_img_file: fh.UploadFile,
    ):
        curr_user = get_curr_user(session)
        res = validate_image_file(profile_img_file)
        if "error" in res.keys():
            fh.add_toast(session, res["error"], "error")
            return (
                fh.Img(
                    src=curr_user.profile_img,
                    cls=f"w-12 h-12 object-cover rounded-full cursor-pointer hover:{img_hover}",
                    id="profile-image-preview",
                ),
            )
        return (
            fh.Img(
                src=f"data:image/png;base64,{res['success']}",
                cls=f"w-12 h-12 object-cover rounded-full cursor-pointer hover:{img_hover}",
                id="profile-image-preview",
            ),
        )

    @f_app.patch("/settings/save")
    def save_settings(
        session,
        email: str | None = None,
        username: str | None = None,
        password: str | None = None,
        profile_img_file: fh.UploadFile | None = None,
    ):
        if not email:
            fh.add_toast(session, "Email is required", "error")
            return None
        if not username:
            fh.add_toast(session, "Username is required", "error")
            return None
        curr_user = get_curr_user(session)
        with get_db_session() as db_session:
            if email != curr_user.email:
                query = select(User).where(User.email == email)
                db_user = db_session.exec(query).first()
                if db_user:
                    fh.add_toast(session, "Email already exists", "error")
                    return None

            if username != curr_user.username:
                query = select(User).where(User.username == username)
                db_user = db_session.exec(query).first()
                if db_user:
                    fh.add_toast(session, "Username already exists", "error")
                    return None

            curr_user.email = email
            curr_user.username = username
            if curr_user.login_type == "email":
                if not password:
                    fh.add_toast(session, "Password is required", "error")
                    return None
                curr_user.hashed_password = pbkdf2_sha256.hash(password)
            if profile_img_file is not None and not profile_img_file.filename == "":
                res = validate_image_file(profile_img_file)
                if "error" in res.keys():
                    fh.add_toast(session, res["error"], "error")
                    return None
                curr_user.profile_img = f"data:image/png;base64,{res['success']}"

            db_session.add(curr_user)
            db_session.commit()
            db_session.refresh(curr_user)
        return fh.Redirect("/settings")

    @f_app.delete("/settings/delete-account")
    def delete_account(session):
        curr_user = get_curr_user(session)
        if curr_user is None:
            return fh.Redirect("/")
        with get_db_session() as db_session:
            db_session.delete(curr_user)
            db_session.commit()
        session.clear()
        return fh.Redirect("/")

    ## stripe
    ### send the user here to buy credits
    @f_app.get("/buy_global")
    def buy_credits():
        s = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": 100,  # Amount in cents, $1.00
                        "product_data": {
                            "name": "Buy 5 credits for $1 (to share)",
                        },
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=DOMAIN + "/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=DOMAIN + "/cancel",
        )
        ### send the USER to STRIPE
        return fh.RedirectResponse(s["url"])

    ### STRIPE sends the USER here after a payment was canceled.
    @f_app.get("/cancel")
    def cancel():
        return fh.RedirectResponse("/")

    ### STRIPE sends the USER here after a payment was 'successful'.
    @f_app.get("/success")
    def success():
        return fh.RedirectResponse("/")

    ### STRIPE calls this to tell APP when a payment was completed.
    @f_app.post("/webhook")
    async def stripe_webhook(
        request,
    ):
        # print(request)
        # print("Received webhook")
        payload = await request.body()
        payload = payload.decode("utf-8")
        signature = request.headers.get("stripe-signature")
        # print(payload)

        # verify the Stripe webhook signature
        try:
            event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
        except ValueError:
            # print("Invalid payload")
            return {"error": "Invalid payload"}, 400
        except stripe.error.SignatureVerificationError:
            # print("Invalid signature")
            return {"error": "Invalid signature"}, 400

        # handle the event
        if event["type"] == "checkout.session.completed":
            # session = event["data"]["object"]
            # print("Session completed", session)
            curr_balance = get_curr_balance()
            curr_balance.balance += 5
            with get_db_session() as db_session:
                db_session.add(curr_balance)
                db_session.commit()
                db_session.refresh(curr_balance)
            return {"status": "success"}, 200

    ## misc
    ### for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    def static_files(fname: str, ext: str):
        static_file_path = PARENT_PATH / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ### toasts without target
    @f_app.post("/toast")
    def toast(session, message: str, type: str):
        fh.add_toast(session, message, type)
        return toast_container()

    return f_app


f_app = get_app()

# -----------------------------------------------------------------------------


@app.function(
    image=IMAGE,
    cpu=CPU,
    memory=MEM,
    secrets=SECRETS,
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():
    return f_app


if __name__ == "__main__":
    fh.serve(app="f_app")
