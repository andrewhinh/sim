import base64
import io
import json
import os
import smtplib
import ssl
import subprocess
import tempfile
from asyncio import sleep
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from uuid import uuid4

import modal

from db.models import (
    DataPoint,
    GlobalBalance,
    Paper,
    SearchParams,
    Trial,
    User,
    init_balance,
)
from src.llm import app as llm_app
from src.llm import (
    check_question_threaded,
    download_models,
    extract_data_points_threaded,
    modify_question_threaded,
    question_to_query_and_papers_threaded,
    rerank_papers_threaded,
)
from utils import (
    APP_NAME,
    CPU,
    MEM,
    MINUTES,
    PARENT_PATH,
    PYTHON_VERSION,
    SECRETS,
)

# -----------------------------------------------------------------------------

# Modal
FE_IMAGE = (
    modal.Image.debian_slim(PYTHON_VERSION)
    .apt_install("git", "libpq-dev")  # add system dependencies
    .pip_install(
        "alembic>=1.15.2",
        "passlib>=1.7.4",
        "pillow>=11.1.0",
        "psycopg2>=2.9.10",
        "python-fasthtml>=0.12.9",
        "simpleicons>=7.21.0",
        "sqlmodel>=0.0.24",
        "starlette>=0.46.1",
        "stripe>=11.6.0",
    )  # add Python dependencies
    .run_commands(
        [
            "git clone https://github.com/Len-Stevens/Python-Antivirus.git /root/Python-Antivirus"
        ]
    )
    .add_local_file(PARENT_PATH / "favicon.ico", "/root/favicon.ico")
    .add_local_file(PARENT_PATH / "logo.png", "/root/logo.png")
)

app = modal.App(APP_NAME)
app.include(llm_app)

# -----------------------------------------------------------------------------

with FE_IMAGE.imports():
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


def get_app():  # noqa: C901
    # styles
    font = "font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'"
    text_color = "stone-600"
    text_hover_color = "stone-500"  # for neutral-colored links
    text_button_hover_color = "stone-700"  # for colored buttons
    error_color = "red-400"
    error_hover_color = "red-500"

    img_hover = "opacity-75"

    border_color = "stone-400"
    border_hover_color = "stone-500"

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

    rounded = "rounded-md"
    shadow = "shadow-md"
    shadow_hover = "shadow-sm"

    click_button = f"{shadow} hover:{shadow_hover} text-{text_color} hover:text-{text_button_hover_color} bg-{click_color} hover:bg-{click_hover_color} border border-{click_border_color} hover:border-{click_border_hover_color}"
    click_neutral_button = f"{shadow} hover:{shadow_hover} text-{text_color} hover:text-{text_hover_color} bg-{click_neutral_color} hover:bg-{click_neutral_hover_color} border border-{click_neutral_border_color} hover:border-{click_neutral_border_hover_color}"
    click_danger_button = f"{shadow} hover:{shadow_hover} text-{text_color} hover:text-{text_button_hover_color} bg-{click_danger_color} hover:bg-{click_danger_hover_color} border border-{click_danger_border_color} hover:border-{click_danger_border_hover_color}"

    input_bg_color = "stone-50"
    input_cls = f"bg-{input_bg_color} {rounded} {shadow} hover:{shadow_hover} text-{text_color} border border-{border_color} hover:border-{border_hover_color}"

    background_color = "stone-200"
    main_page = f"flex flex-col justify-between min-h-screen w-full bg-{background_color} text-{text_color} {font}"
    page_ctnt = "flex flex-col grow gap-4 p-8"

    tailwind_to_hex = {
        click_color: "#60A5FA",
        text_color: "#57534E",
    }
    font_hex = (
        font.split(":")[-1].split(",")[0].strip("'").strip('"')
    )  # Extract just the font family without the CSS property part

    # FastHTML setup
    def before(req, sess):
        if "session_uuid" not in sess:
            req.scope["session_uuid"] = sess.setdefault(
                "session_uuid", str(uuid4())
            )  # must be json-serializable
        if "user_uuid" not in sess:
            req.scope["user_uuid"] = sess.setdefault("user_uuid", "")

    def _not_found(req, exc):
        return (
            fh.Title(APP_NAME + " | 404"),
            fh.Div(
                nav(req.session),
                fh.Main(
                    fh.P(
                        "Page not found!",
                        cls=f"text-2xl text-{error_color} hover:text-{error_hover_color}",
                    ),
                    cls=f"{page_ctnt} justify-center items-center",
                ),
                toast_container(),
                cls=main_page,
            ),
        )

    f_app, _ = fh.fast_app(
        exts="ws",
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
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
            fh.Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
            fh.Style("""
            .hide-when-loading {
                display: inline-block;
            }
            .htmx-request.hide-when-loading,
            .htmx-request .hide-when-loading {
                display: none;
            }
            .indicator {
                display: none;
            }
            .htmx-request.indicator,
            .htmx-request .indicator {
                display: inline-block;
            }
            """),
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
    engine = create_engine(url=os.getenv("DATABASE_URL"), echo=False)

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

    def get_curr_balance() -> GlobalBalance:
        with get_db_session() as db_session:
            curr_balance = db_session.get(GlobalBalance, 1)
            if not curr_balance:
                curr_balance = GlobalBalance(balance=init_balance)
                db_session.add(curr_balance)
                db_session.commit()
                db_session.refresh(curr_balance)
            return curr_balance

    def get_curr_user(session) -> User | None:
        if session["user_uuid"]:
            with get_db_session() as db_session:
                query = select(User).where(User.uuid == session["user_uuid"])
                curr_user = db_session.exec(query).first()
                if not curr_user:
                    return None
                return curr_user
        return None

    def get_curr_trials(session) -> list[Trial]:
        if session["user_uuid"]:
            with get_db_session() as db_session:
                query = select(User).where(User.uuid == session["user_uuid"])
                curr_user = db_session.exec(query).first()
                if not curr_user:
                    return []
                return curr_user.trials
        return []

    def get_trial(session, uuid: str) -> Trial | None:
        with get_db_session() as db_session:
            query = (
                select(Trial)
                .where(Trial.uuid == uuid)
                .where(Trial.session_uuid == session["session_uuid"])
            )
            curr_trial = db_session.exec(query).first()
            if not curr_trial:
                return None
            return curr_trial

    # stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    DOMAIN = os.getenv("DOMAIN")

    # SSE state
    shutdown_event = fh.signal_shutdown()
    global shown_balance
    shown_balance = 0
    global shown_trials
    shown_trials = {}

    # OAuth
    google_client = GoogleAppClient(
        os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET")
    )
    github_client = GitHubAppClient(
        os.getenv("GITHUB_CLIENT_ID"), os.getenv("GITHUB_CLIENT_SECRET")
    )

    # constants
    phrase_to_question = {
        "keto and gum disease": "What is the relationship between adherence to a ketogenic diet and the prevalence of periodontal disease among adults aged 30-50?",
        "fasting and cognition": "How does intermittent fasting affect cognitive performance and memory function in healthy adults over a 12-week period?",
        "CV health and exercise intensity": "To what extent does high-intensity interval training versus moderate continuous exercise impact cardiovascular health markers in sedentary middle-aged individuals?",
    }
    sections = [
        "Searching for papers",
        "Screening papers",
        "Extracting data",
        "Generating participants",
        "Simulating study",
        "Clustering participants",
        "Generating visualization",
    ]

    # ui components
    def spinner(id="", cls=""):
        return (
            fh.Svg(
                fh.NotStr(
                    """<svg class="animate-spin" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" />
                        <path class="opacity-75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" fill="currentColor" />
                    </svg>"""
                ),
                id=id,
                cls=f"animate-spin {cls}",
            ),
        )

    ## trial manage page
    def trial_view(
        trial: Trial,
    ):
        limit_chars = 1000
        return (
            fh.Tr(
                fh.Td(
                    fh.Div(
                        fh.Input(
                            type="checkbox",
                            name="selected_trials",
                            value=trial.uuid,
                            hx_target="#trial-manage",
                            hx_swap="outerHTML",
                            hx_trigger="change",
                            hx_post="/trials/show-select-delete",
                            hx_indicator="#spinner",
                            cls="hide-when-loading",
                        ),
                        spinner(
                            id=f"trial-{trial.uuid}-show-select-delete-loader",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        fh.Svg(
                            fh.NotStr(
                                """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z" /></svg>"""
                            ),
                            hx_delete=f"/trials/{trial.uuid}",
                            hx_indicator="#spinner",
                            hx_target="closest tr",
                            hx_swap="outerHTML swap:.25s",
                            hx_confirm="Are you sure?",
                            cls=f"hide-when-loading size-8 text-{click_danger_color} hover:text-{click_danger_hover_color}",
                        ),
                        spinner(
                            id=f"trial-{trial.uuid}-delete-loader",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        cls="w-1/6 flex justify-start items-center gap-2",
                    ),
                    fh.Div(
                        fh.A(
                            trial.question[:limit_chars] + "..."
                            if len(trial.question) > limit_chars
                            else trial.question,
                            href=f"/trials/{trial.uuid}",
                            cls=f"text-{text_color} hover:text-{text_hover_color}",
                        ),
                        spinner(
                            id=f"trial-{trial.uuid}-main-loader",
                            hx_ext="sse",
                            sse_connect=f"/trials/{trial.uuid}/stream",
                            sse_swap="UpdateTrial",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        cls="w-5/6",
                    ),
                    id=f"trial-{trial.uuid}",
                    cls="w-full flex justify-between items-center p-4",
                ),
                cls="flex grow",
            ),
        )

    def num_trials(session, hx_swap_oob: str = "false"):
        return fh.Div(
            fh.P(
                f"({len(get_curr_trials(session))} total trials)",
                cls=f"text-{text_color}",
            ),
            id="trial-count",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="flex justify-center items-center",
        )

    def trial_manage(
        session, trials_selected: bool = False, hx_swap_oob: str = "false"
    ):
        trials_present = len(get_curr_trials(session)) > 0
        return fh.Div(
            fh.Button(
                fh.P(
                    "Delete selected",
                    id="delete-select-trials-button-text",
                    cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                ),
                spinner(
                    id="delete-select-trials-loader",
                    cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                ),
                hx_delete="/trials/select",
                hx_target="#trial-list",
                hx_confirm="Are you sure?",
                hx_indicator="#delete-select-trials-button-text, #delete-select-trials-loader",
                hx_disabled_elt="#delete-select-trials-button-text, #delete-select-trials-loader",
                cls=f"w-full flex justify-center items-center {click_danger_button} {rounded} p-3",
            )
            if trials_present and trials_selected
            else None,
            fh.Button(
                fh.P(
                    "Delete all",
                    id="delete-all-trials-button-text",
                    cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                ),
                spinner(
                    id="delete-all-trials-loader",
                    cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                ),
                hx_delete="/trials",
                hx_target="#trial-list",
                hx_confirm="Are you sure?",
                hx_indicator="#delete-all-trials-button-text, #delete-all-trials-loader",
                hx_disabled_elt="#delete-all-trials-button-text, #delete-all-trials-loader",
                cls=f"w-full flex justify-center items-center {click_danger_button} {rounded} p-3",
            )
            if trials_present
            else None,
            id="trial-manage",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="flex flex-col md:flex-row justify-center items-center gap-4 w-full",
        )

    def get_trial_table_part(session, part_num: int = 1):
        size = 50
        curr_trials = get_curr_trials(session)
        if not curr_trials:
            return ()
        curr_trials_part = curr_trials[(part_num - 1) * size : part_num * size]
        global shown_trials
        shown_trials = list(
            set(shown_trials) | {trial.id for trial in curr_trials_part}
        )
        paginated = [trial_view(trial) for trial in curr_trials_part]
        return tuple(paginated)

    def trial_load_more(session, idx: int = 2, hx_swap_oob: str = "false"):
        n_trials = len(get_curr_trials(session))
        trials_present = n_trials > 0
        global shown_trials
        still_more = n_trials > len(shown_trials)
        return fh.Div(
            fh.Button(
                fh.P(
                    "Load More",
                    id="load-more-trials-button-text",
                    cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                ),
                spinner(
                    id="load-more-trials-loader",
                    cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                ),
                hx_get=f"/trials/page/{idx}",
                hx_indicator="#load-more-trials-button-text, #load-more-trials-loader",
                hx_target="#trial-list",
                hx_swap="beforeend",
                hx_disabled_elt="#load-more-trials-button-text, #load-more-trials-loader",
                cls=f"w-full flex justify-center items-center {click_button} {rounded} p-3",
            )
            if trials_present and still_more
            else None,
            id="load-more-trials",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-full md:w-2/3",
        )

    ## single trial

    def section_view(
        section: str,
        status: str = "pending",
        section_id: str = "",
    ):
        status_elements = {
            "pending": fh.Svg(
                fh.NotStr(
                    """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" fill="none"/>
                    </svg>""",
                ),
                cls=f"size-6 text-{text_color}",
            ),
            "running": spinner(
                cls=f"size-6 text-{text_color} hover:text-{text_button_hover_color}",
            ),
            "completed": fh.Svg(
                fh.NotStr(
                    """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" fill="currentColor"/>
                    </svg>"""
                ),
                cls=f"size-6 text-{click_color}",
            ),
            "failed": fh.Svg(
                fh.NotStr(
                    """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" fill="currentColor"/>
                    </svg>"""
                ),
                cls=f"size-6 text-{error_color}",
            ),
        }
        return fh.Div(
            status_elements.get(status, status_elements["pending"]),
            fh.P(
                section,
                cls=f"text-md text-{text_color}",
            ),
            id=section_id,
            cls="w-full md:w-2/3 flex justify-start items-center gap-12 md:gap-16",
        )

    def trial_expanded(trial: Trial):
        return (
            fh.Div(
                id=f"trial-visualization-{trial.uuid}",
            )
            if trial.success is True
            else fh.Div(
                *[
                    section_view(
                        section,
                        "pending",
                        f"section-{trial.uuid}-{i}",
                    )
                    for i, section in enumerate(sections)
                ],
                id=f"trial-visualization-{trial.uuid}",
                hx_ext="ws",
                ws_connect=f"/trial/ws/{trial.uuid}",
                hx_trigger="load",
                cls="w-full md:w-2/3 h-2/3 flex flex-col grow justify-start items-end pt-8 gap-8 md:gap-12",
            ),
        )

    # ui layout
    def nav(session, show_auth=True):
        curr_user = get_curr_user(session)
        return fh.Nav(
            fh.Div(
                fh.A(
                    fh.Img(
                        src="/logo.png",
                        cls="size-10 object-contain",
                    ),
                    fh.P(
                        APP_NAME,
                        cls=f"text-lg text-{text_color} ",
                    ),
                    href="/",
                    cls=f"flex justify-center items-center gap-2 hover:{img_hover} hover:text-{text_hover_color}",
                ),
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
                    id="question",
                    placeholder="Ask a research question...",
                    rows=10,
                    autofocus=True,
                    hx_post="/trial/check-question",
                    hx_target="#check-results",
                    hx_swap="outerHTML",
                    hx_trigger="change, keyup delay:200ms changed",
                    hx_indicator="#check-results, #check-loader",
                    hx_disabled_elt="#question-form-button",
                    cls=f"{input_cls} p-4 resize-none",
                ),
                fh.Div(id="new-textarea-content", cls="hidden"),
                fh.Div(
                    fh.Div(
                        id="check-results",
                        hx_swap_oob="true",
                        cls="hide-when-loading",
                    ),
                    fh.Div(
                        spinner(
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_hover_color}"
                        ),
                        fh.P(
                            "Evaluating question strength...",
                            cls=f"indicator text-{text_color} hover:text-{text_hover_color}",
                        ),
                        id="check-loader",
                        cls="indicator flex justify-center items-center space-x-2",
                    ),
                    fh.Div(
                        spinner(
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_hover_color}"
                        ),
                        fh.P(
                            "Modifying question...",
                            cls=f"indicator text-{text_color} hover:text-{text_hover_color}",
                        ),
                        id="modify-loader",
                        cls="indicator flex justify-center items-center space-x-2",
                    ),
                    cls="absolute bottom-2 left-2 p-2",
                ),
                fh.Button(
                    fh.P(
                        "→",
                        id="question-form-button-text",
                        cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    spinner(
                        id="question-form-loader",
                        cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    id="question-form-button",
                    type="submit",
                    cls=f"absolute bottom-0 right-4 rounded-full p-2 {click_button}",
                    style="width: 3rem; height: 3rem;",
                ),
                id="question-form",
                hx_post="/trial",
                hx_target="this",
                hx_swap="none",
                hx_trigger="keyup[shiftKey&&key=='Enter'] from:body, click from:#question-form-button",
                hx_indicator="#question-form-button-text, #question-form-loader",
                hx_disabled_elt="#question, #question-form-button",
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
                        question,
                        cls=f"{click_button} px-4 py-2 {rounded}",
                        hx_post="/trial/fill-question",
                        hx_target="#new-textarea-content",
                        hx_vals=json.dumps(
                            {
                                "question": question,
                            }
                        ),
                    )
                    for question in phrase_to_question.keys()
                ],
                cls="w-full md:w-2/3 flex flex-col md:flex-row justify-start items-start md:items-center gap-4",
            ),
            cls=f"{page_ctnt} justify-center items-center",
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
                                cls="size-6",
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
                                cls="size-6",
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
                        fh.P(
                            "Sign Up",
                            id="signup-button-text",
                            cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        spinner(
                            id="signup-loader",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        id="signup-button",
                        type="submit",
                        cls=f"w-full {click_button} p-3 {rounded}",
                    ),
                    hx_post="/auth/signup",
                    hx_target="this",
                    hx_swap="none",
                    hx_indicator="#signup-button-text, #signup-loader",
                    hx_disabled_elt="#email, #password, #signup-button",
                    cls="w-full",
                ),
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=f"{page_ctnt} justify-center items-center",
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
                                cls="size-6",
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
                                cls="size-6",
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
                        fh.P(
                            "Log In",
                            id="login-button-text",
                            cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        spinner(
                            id="login-loader",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        id="login-button",
                        type="submit",
                        cls=f"w-full {click_button} p-3 {rounded}",
                    ),
                    hx_post="/auth/login",
                    hx_target="this",
                    hx_swap="none",
                    hx_indicator="#login-button-text, #login-loader",
                    hx_disabled_elt="#email, #password, #login-button",
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
            cls=f"{page_ctnt} justify-center items-center",
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
                    id="email",
                    type="email",
                    name="email",
                    placeholder="Email",
                    required=True,
                    cls=f"w-full {input_cls}",
                ),
                fh.Button(
                    fh.P(
                        "Send Reset Link",
                        id="forgot-password-button-text",
                        cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    spinner(
                        id="forgot-password-loader",
                        cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    id="forgot-password-button",
                    type="submit",
                    cls=f"w-full {click_button} p-3 {rounded}",
                ),
                hx_post="/auth/forgot-password",
                hx_target="this",
                hx_swap="none",
                hx_indicator="#forgot-password-button-text, #forgot-password-loader",
                hx_disabled_elt="#email, #forgot-password-button",
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=f"{page_ctnt} justify-center items-center",
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
                    id="password",
                    name="password",
                    placeholder="New Password",
                    type="password",
                    required=True,
                    cls=f"w-full {input_cls}",
                ),
                fh.Input(
                    id="confirm_password",
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
                    fh.P(
                        "Reset Password",
                        id="reset-password-button-text",
                        cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    spinner(
                        id="reset-password-loader",
                        cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    id="reset-password-button",
                    type="submit",
                    cls=f"w-full {click_button} p-3 {rounded}",
                ),
                hx_post="/auth/reset-password",
                hx_target="this",
                hx_swap="none",
                hx_indicator="#reset-password-button-text, #reset-password-loader",
                hx_disabled_elt="#password, #confirm_password, #reset-password-button",
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-4 {input_cls} p-8",
            ),
            cls=f"{page_ctnt} justify-center items-center",
        )

    def trials_content(session):
        return fh.Main(
            fh.Div(
                fh.H1(
                    "Trials", cls=f"text-4xl font-bold text-{text_color} text-center"
                ),
                fh.Div(
                    num_trials(session),
                    trial_manage(session),
                    get_trial_table_part(session),
                    trial_load_more(session),
                    cls="w-full flex flex-col justify-center items-center gap-4",
                ),
                cls="w-full md:w-2/3 flex flex-col justify-center items-center gap-8",
            ),
            cls=f"{page_ctnt} justify-start items-center",
        )

    def trial_content(session, uuid: str):
        trial = get_trial(session, uuid)
        if not trial:
            return fh.Main(
                fh.P(
                    "Trial not found!",
                    cls=f"text-2xl text-{error_color} hover:text-{error_hover_color}",
                ),
                cls=f"{page_ctnt} justify-center items-center",
            )
        return fh.Main(
            fh.Div(
                fh.H1(
                    trial.question,
                    cls=f"text-lg italic text-{text_color} text-center",
                ),
                cls="w-full md:w-2/3 h-1/3 flex flex-col grow justify-center items-center",
            ),
            trial_expanded(trial),
            cls=f"{page_ctnt} justify-start items-center",
        )

    def settings_content(session):
        curr_user = get_curr_user(session)
        if not curr_user:
            return fh.Main(
                fh.P(
                    "User not found!",
                    cls=f"text-2xl text-{error_color} hover:text-{error_hover_color}",
                ),
                cls=f"{page_ctnt} justify-center items-center",
            )
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
                                cls="size-12 object-cover rounded-full",
                            ),
                            fh.Button(
                                fh.P(
                                    "Edit",
                                    id="edit-button-text",
                                    cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                                ),
                                spinner(
                                    id="edit-loader",
                                    cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                                ),
                                hx_get="/user/settings/edit",
                                hx_target="#settings",
                                hx_swap="outerHTML",
                                hx_indicator="#edit-button-text, #edit-loader",
                                hx_disabled_elt="#edit-button",
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
                        fh.P(
                            "Delete Account",
                            id="delete-account-button-text",
                            cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        spinner(
                            id="delete-account-loader",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        hx_delete="/user/settings/delete-account",
                        hx_confirm="Are you sure you want to delete your account? This action cannot be undone.",
                        hx_indicator="#delete-account-button-text, #delete-account-loader",
                        hx_disabled_elt="#delete-account-button",
                        cls=f"w-full flex justify-center items-center {click_danger_button} {rounded} p-3",
                    ),
                    cls="w-full flex flex-col justify-center items-center gap-4",
                ),
                cls=f"w-full md:w-1/3 flex flex-col justify-center items-center gap-8 {input_cls} p-8",
            ),
            cls=f"{page_ctnt} justify-center items-center",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        curr_balance = get_curr_balance()
        return fh.Footer(
            fh.Div(
                fh.Div(
                    fh.P("Global balance:", cls=f"text-{text_color}"),
                    fh.P(
                        f"{curr_balance.balance} credits",
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
                        cls=f"size-10 object-contain hover:{img_hover}",
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
            curr_balance = get_curr_balance()
            global shown_balance
            if shown_balance != curr_balance.balance:
                shown_balance = curr_balance.balance
                yield f"""event: UpdateBalance\ndata: {fh.to_xml(fh.P(f"{shown_balance} credits", cls=f"text-{text_color} font-bold", sse_swap="UpdateBalance"))}\n\n"""
            await sleep(1)

    async def stream_trial_updates(
        session,
        uuid: str,
    ):
        while not shutdown_event.is_set():
            trial = get_trial(session, uuid)
            if trial:
                curr_state = (
                    "success"
                    if trial.success is True
                    else "failed"
                    if trial.success is False
                    else "loading"
                )
                global shown_trials
                if shown_trials.get(uuid) != curr_state:
                    shown_trials[uuid] = curr_state
                    yield f"""event: UpdateTrials\ndata: {fh.to_xml(
                    spinner(
                            id=f"trial-{trial.uuid}-main-loader",
                            sse_swap="UpdateTrials",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                    ) if curr_state == "loading" else
                    fh.Div(
                        fh.Svg(
                            fh.NotStr(
                                """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" /></svg>"""
                            ),
                            cls=f"size-6 text-{error_color} hover:text-{error_hover_color}",
                        ),
                        sse_swap="UpdateTrials",
                    ) if curr_state == "failed" else
                    fh.Div(
                        sse_swap="UpdateTrials",
                    ))}\n\n"""
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
        sender_email = os.getenv("EMAIL_SENDER")
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT"))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")

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
                trials_content(session),
                toast_container(),
                cls=main_page,
            ),
        )

    @f_app.get("/trial/{uuid}")
    def trial_page(session, uuid: str):
        return (
            fh.Title(f"{APP_NAME} | trial"),
            fh.Div(
                nav(session, True),
                trial_content(session, uuid),
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
    @f_app.get("/stream-balance")
    async def stream_balance():
        """Stream balance updates to connected clients"""
        return StreamingResponse(
            stream_balance_updates(), media_type="text/event-stream"
        )

    ## overlay
    @f_app.get("/user/overlay/show")
    def overlay_show(session):
        curr_user = get_curr_user(session)
        if not curr_user:
            fh.add_toast(session, "Refresh page", "error")
            return
        max_username_length = 17
        return fh.Div(
            fh.Div(
                fh.Img(
                    src=curr_user.profile_img,
                    cls=f"size-10 object-cover hover:{img_hover} rounded-full",
                ),
                fh.P(
                    curr_user.username
                    if len(curr_user.username) <= max_username_length
                    else curr_user.username[:max_username_length] + "...",
                    cls=f"text-{text_color} hover:text-{text_hover_color} hidden md:block",
                ),
                hx_get="/user/overlay/close",
                hx_target="#overlay",
                hx_swap="outerHTML",
                hx_trigger="click",
                hx_disabled_elt="#overlay",
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
                    fh.P(
                        "Log out",
                        id="logout-button-text",
                        cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    spinner(
                        id="logout-loader",
                        cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                    ),
                    cls=f"w-full {click_button} {rounded} p-3",
                    hx_get="/auth/logout",
                    hx_indicator="#logout-button-text, #logout-loader",
                    hx_disabled_elt="#logout-button",
                ),
                cls=f"absolute top-14 right-0 min-w-40 z-10 flex flex-col justify-center items-center gap-2 {input_cls} p-2",
            ),
            id="overlay",
            cls="relative",
        )

    @f_app.get("/user/overlay/close")
    def overlay_close(session):
        curr_user = get_curr_user(session)
        if not curr_user:
            fh.add_toast(session, "Refresh page", "error")
            return
        max_username_length = 17
        return fh.Div(
            fh.Img(
                src=curr_user.profile_img,
                cls=f"size-10 object-cover hover:{img_hover} rounded-full",
            ),
            fh.P(
                curr_user.username
                if len(curr_user.username) <= max_username_length
                else curr_user.username[:max_username_length] + "...",
                cls=f"text-{text_color} hover:text-{text_hover_color} hidden md:block",
            ),
            id="overlay",
            hx_get="/user/overlay/show",
            hx_target="#overlay",
            hx_swap="outerHTML",
            hx_trigger="click",
            hx_disabled_elt="#overlay",
            cls=f"flex justify-center items-center gap-2 cursor-pointer hover:{img_hover} hover:text-{text_hover_color}",
        )

    ## create trial
    @f_app.post("/trial/fill-question")
    def fill_question(session, question: str):
        return fh.Script(f"""
            document.getElementById('question').value = "{phrase_to_question.get(question)}";
        """)

    @f_app.post("/trial/check-question")
    def check_question(session, question: str):
        suggestions = (
            check_question_threaded.local(question)
            if modal.is_local()
            else check_question_threaded.remote(question)
        )
        return (
            fh.Div(
                "Good question!",
                id="check-results",
                hx_swap_oob="true",
                cls=f"hide-when-loading text-{click_color} hover:text-{click_hover_color}",
            )
            if not suggestions
            else fh.Div(
                fh.P(
                    "Suggestions:",
                    cls=f"text-{text_color} hover:text-{text_hover_color}",
                ),
                *[
                    fh.Button(
                        suggestion,
                        hx_post="/trial/modify-question",
                        hx_target="#new-textarea-content",
                        hx_indicator="#check-results, #modify-loader",
                        hx_disabled_elt="#question-form-button",
                        hx_vals=json.dumps(
                            {
                                "question": question,
                                "suggestion": suggestion,
                            }
                        ),
                        cls=f"{click_neutral_button} px-4 py-2 {rounded}",
                    )
                    for suggestion in suggestions
                ],
                id="check-results",
                hx_swap_oob="true",
                cls="hide-when-loading w-full md:w-2/3 flex flex-col md:flex-row justify-start items-start md:items-center gap-4",
            ),
        )

    @f_app.post("/trial/modify-question")
    def modify_question(session, question: str, suggestion: str):
        new_question = (
            modify_question_threaded.local(question, suggestion)
            if modal.is_local()
            else modify_question_threaded.remote(question, suggestion)
        )
        return (
            fh.Script(f"""
                document.getElementById('question').value = "{new_question}";
            """),
            fh.Div(
                id="check-results",
                hx_swap_oob="true",
                cls="hide-when-loading",
            ),
        )

    @f_app.post("/trial")
    def sim_trial(session, question: str):
        with get_db_session() as db_session:
            trial = Trial(
                question=question,
                session_uuid=session["session_uuid"],
                user=get_curr_user(session),
            )
            db_session.add(trial)
            db_session.commit()
            db_session.refresh(trial)
        return fh.Redirect(f"/trial/{trial.uuid}")

    async def on_connect(session, ws, send):
        uuid = ws.path_params["uuid"]
        trial = get_trial(session, uuid)

        # on fail
        async def on_fail(trial: Trial):
            with get_db_session() as db_session:
                trial.success = False
                db_session.add(trial)
                db_session.commit()
                db_session.refresh(trial)
            await send(trial_expanded(trial))
            await ws.close()
            return

        # create search parameters and search papers
        await send(
            section_view(sections[0], "running", f"section-{uuid}-0"),
        )
        search_params, papers = (
            question_to_query_and_papers_threaded.local(trial.question)
            if modal.is_local()
            else question_to_query_and_papers_threaded.remote(trial.question)
        )
        status = "completed" if search_params and papers else "failed"
        await send(
            section_view(sections[0], status, f"section-{uuid}-0"),
        )
        if status == "completed":
            with get_db_session() as db_session:
                db_search_params = SearchParams(**search_params, trial=trial)
                db_session.add(db_search_params)

                db_papers = []
                for paper in papers:
                    paper = Paper(**paper)
                    db_papers.append(paper)
                db_session.add(trial)
                trial.papers = db_papers

                db_session.commit()
                db_session.refresh(db_search_params)
                db_session.refresh(trial)
        else:
            return await on_fail(trial)

        # screen papers
        await send(
            section_view(sections[1], "running", f"section-{uuid}-1"),
        )
        papers = (
            rerank_papers_threaded.local(trial.question, papers)
            if modal.is_local()
            else rerank_papers_threaded.remote(trial.question, papers)
        )
        status = "completed" if papers else "failed"
        await send(
            section_view(sections[1], status, f"section-{uuid}-1"),
        )
        if status == "completed":
            with get_db_session() as db_session:
                db_papers = []
                for paper in papers:
                    db_paper = Paper(**paper)
                    db_papers.append(db_paper)
                db_session.add(trial)
                trial.papers = db_papers
                db_session.commit()
                db_session.refresh(trial)
        else:
            return await on_fail(trial)

        # extract data points
        await send(
            section_view(sections[2], "running", f"section-{uuid}-2"),
        )
        data_points = (
            extract_data_points_threaded.local(papers)
            if modal.is_local()
            else extract_data_points_threaded.remote(papers)
        )
        status = "completed" if data_points else "failed"
        await send(
            section_view(sections[2], status, f"section-{uuid}-2"),
        )
        if status == "completed":
            with get_db_session() as db_session:
                for paper, data_points in zip(papers, data_points):
                    db_data_points = []
                    for data_point in data_points:
                        db_data_point = DataPoint(**data_point)
                        db_data_points.append(db_data_point)
                    db_session.add(paper)
                    paper.data_points = db_data_points
                db_session.add(trial)
                trial.papers = papers
                db_session.commit()
                db_session.refresh(trial)
        else:
            return await on_fail(trial)

        # success
        with get_db_session() as db_session:
            trial.success = True
            db_session.add(trial)
            db_session.commit()
            db_session.refresh(trial)
        await send(trial_expanded(trial))
        await ws.close()
        return

    async def on_disconnect():
        pass

    @f_app.ws("/trial/ws/{uuid}", conn=on_connect, disconn=on_disconnect)
    async def trial_progress_ws():
        pass

    ## manage trials
    @f_app.get("/trials/{uuid}/stream")
    async def stream_trial(session, uuid: str):
        return StreamingResponse(
            stream_trial_updates(session, uuid), media_type="text/event-stream"
        )

    @f_app.get("/trials/page/{idx}")
    def page_gens(session, idx: int):
        part = get_trial_table_part(session, idx)  # to modify global shown_generations
        return part, trial_load_more(
            session,
            idx + 1,
            "true",
        )

    @f_app.delete("/trials")
    def delete_trials(
        session,
    ):
        trials = get_curr_trials(session)
        global shown_trials
        for t in trials:
            with get_db_session() as db_session:
                db_session.delete(t)
                db_session.commit()
            shown_trials.pop(t.uuid, None)
        fh.add_toast(session, "Deleted trials.", "success")
        return (
            "",
            num_trials(session, "true"),
            trial_manage(
                session,
                hx_swap_oob="true",
            ),
            trial_load_more(
                session,
                hx_swap_oob="true",
            ),
        )

    @f_app.delete("/trials/select")
    def delete_select_trials(session, selected_trials: list[str] = None):
        if selected_trials:
            curr_trials = get_curr_trials(session)
            global shown_trials
            select_trials = [
                trial for trial in curr_trials if trial.uuid in selected_trials
            ]
            with get_db_session() as db_session:
                for t in select_trials:
                    db_session.delete(t)
                    db_session.commit()
                    shown_trials.pop(t.uuid, None)
            fh.add_toast(session, "Deleted trials.", "success")
            remain_trials = get_curr_trials(session)
            remain_view = [trial_view(t, session) for t in remain_trials[::-1]]
            return (
                remain_view,
                num_trials(session, "true"),
                trial_manage(
                    session,
                    hx_swap_oob="true",
                ),
                trial_load_more(
                    session,
                    hx_swap_oob="true",
                ),
            )
        else:
            fh.add_toast(session, "No generations selected.", "warning")
            return fh.Response(status_code=204)

    @f_app.post("/trials/show-select-delete")
    def show_select_trial_delete(session, selected_trials: list[str] = None):
        return trial_manage(
            session,
            len(selected_trials) > 0,
        )

    @f_app.delete("/trials/{trial_uuid}")
    def delete_trial(
        session,
        trial_uuid: str,
    ):
        trial = get_trial(session, trial_uuid)
        if not trial:
            return
        with get_db_session() as db_session:
            db_session.delete(trial)
            db_session.commit()
        global shown_trials
        shown_trials.pop(trial_uuid, None)
        fh.add_toast(session, "Deleted trial.", "success")
        return (
            "",
            num_trials(session, "true"),
            trial_manage(
                session,
                hx_swap_oob="true",
            ),
            trial_load_more(
                session,
                hx_swap_oob="true",
            ),
        )

    ## auth
    @f_app.get("/auth/logout")
    def logout(session):
        if session.get("user_uuid"):
            del session["user_uuid"]
        return fh.Redirect("/")

    @f_app.post("/auth/login")
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
                return fh.Redirect("/signup")
            else:
                if not pbkdf2_sha256.verify(password, db_user.hashed_password):
                    fh.add_toast(session, "Incorrect credentials", "error")
                    return None
                else:
                    session["user_uuid"] = db_user.uuid
                    return fh.Redirect("/")

    @f_app.post("/auth/signup")
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
                return fh.Redirect("/login")
            else:
                db_user = User(
                    login_type="email",
                    profile_img=f"data:image/svg+xml;base64,{base64.b64encode(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"50\" fill=\"{tailwind_to_hex[click_color]}\"/><text x=\"50\" y=\"60\" font-size=\"40\" text-anchor=\"middle\" fill=\"{tailwind_to_hex[text_color]}\" font-family=\"{font_hex}\">{email[0].upper()}</text></svg>'.encode()).decode()}",
                    email=email,
                    username=email,
                    hashed_password=pbkdf2_sha256.hash(password),
                )
                db_session.add(db_user)
                db_session.commit()
                db_session.refresh(db_user)
                session["user_uuid"] = db_user.uuid
                return fh.Redirect("/")

    @f_app.post("/auth/forgot-password")
    def forgot_password(session, email: str):
        if not email:
            fh.add_toast(session, "Email is required", "error")
            return None

        token_expiry = 24  # hours

        with get_db_session() as db_session:
            query = select(User).where(User.email == email)
            db_user = db_session.exec(query).first()
            if not db_user:
                return fh.Redirect("/login")
            if db_user.login_type != "email":
                fh.add_toast(
                    session, "This account uses a different login method.", "error"
                )
                return None

            reset_token = str(uuid4())
            db_user.reset_token = reset_token
            db_user.reset_token_expiry = datetime.now() + timedelta(hours=token_expiry)
            db_session.add(db_user)
            db_session.commit()
            db_session.refresh(db_user)

        reset_link = f"{os.getenv('DOMAIN')}/reset-password?token={reset_token}"
        send_password_reset_email(email, reset_link)

        return fh.Redirect("/login")

    @f_app.post("/auth/reset-password")
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
            return fh.Redirect("/login")

        with get_db_session() as db_session:
            query = select(User).where(User.reset_token == token)
            db_user = db_session.exec(query).first()
            if not db_user:
                return fh.Redirect("/login")
            if db_user.reset_token_expiry < datetime.now():
                return fh.Redirect("/login")

            db_user.hashed_password = pbkdf2_sha256.hash(password)
            db_user.reset_token = None
            db_user.reset_token_expiry = None
            db_session.add(db_user)
            db_session.commit()
            db_session.refresh(db_user)

        return fh.Redirect("/login")

    @f_app.get("/redirect-github")
    def redirect_github(
        request,
        session,
        code: str | None = None,
        error: str | None = None,
    ):
        if not code or error:
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
                        "profile_img": f"data:image/svg+xml;base64,{base64.b64encode(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"50\" fill=\"{tailwind_to_hex[click_color]}\"/><text x=\"50\" y=\"60\" font-size=\"40\" text-anchor=\"middle\" fill=\"{tailwind_to_hex[text_color]}\" font-family=\"{font_hex}\">{email[0].upper() if email else username[0].upper()}</text></svg>'.encode()).decode()}",
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
    def redirect_google(
        request,
        session,
        code: str | None = None,
        error: str | None = None,
    ):
        if not code or error:
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
                        "profile_img": f"data:image/svg+xml;base64,{base64.b64encode(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"50\" fill=\"{tailwind_to_hex[click_color]}\"/><text x=\"50\" y=\"60\" font-size=\"40\" text-anchor=\"middle\" fill=\"{tailwind_to_hex[text_color]}\" font-family=\"{font_hex}\">{email[0].upper() if email else "U"}</text></svg>'.encode()).decode()}",
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
    @f_app.get("/user/settings/edit")
    def edit_settings(session):
        curr_user = get_curr_user(session)
        if not curr_user:
            fh.add_toast(session, "Refresh page", "error")
            return
        return (
            fh.Form(
                fh.Div(
                    fh.Label(
                        fh.Input(
                            id="new-profile-img-upload",
                            name="profile_img_file",
                            type="file",
                            accept="image/*",
                            hx_post="/user/settings/update-preview",
                            hx_target="#profile-img-preview",
                            hx_swap="outerHTML",
                            hx_trigger="change delay:200ms changed",
                            hx_indicator="#profile-img-preview, #profile-img-loader",
                            hx_disabled_elt="#new-profile-img-upload",
                            hx_encoding="multipart/form-data",
                            cls="hidden",
                        ),
                        fh.Img(
                            src=curr_user.profile_img,
                            cls=f"hide-when-loading size-12 object-cover rounded-full cursor-pointer hover:{img_hover}",
                            id="profile-img-preview",
                        ),
                        spinner(
                            id="profile-img-loader",
                            cls=f"indicator size-12 text-{text_color} hover:text-{text_hover_color}",
                        ),
                    ),
                    fh.Button(
                        fh.P(
                            "Save",
                            id="save-button-text",
                            cls=f"hide-when-loading text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        spinner(
                            id="save-loader",
                            cls=f"indicator size-6 text-{text_color} hover:text-{text_button_hover_color}",
                        ),
                        id="save-button",
                        type="submit",
                        hx_patch="/user/settings/save",
                        hx_target="this",
                        hx_swap="none",
                        hx_indicator="#save-button-text, #save-loader",
                        hx_disabled_elt="#new-profile-img-upload, #email, #username, #password, #save-button",
                        hx_include="#new-profile-img-upload",
                        hx_encoding="multipart/form-data",
                        cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {click_button} {rounded} p-3",
                    ),
                    cls="w-full flex justify-between items-center",
                ),
                fh.Div(
                    fh.P("Email:", cls=f"text-{text_color}"),
                    fh.Input(
                        id="email",
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
                        id="username",
                        value=curr_user.username,
                        name="username",
                        cls=f"max-w-28 md:max-w-48 flex grow justify-center items-center {input_cls}",
                    ),
                    cls="w-full flex justify-between items-center",
                ),
                fh.Div(
                    fh.P("Password:", cls=f"text-{text_color}"),
                    fh.Input(
                        id="password",
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

    @f_app.post("/user/settings/update-preview")
    def update_preview(
        session,
        profile_img_file: fh.UploadFile,
    ):
        curr_user = get_curr_user(session)
        if not curr_user:
            fh.add_toast(session, "Refresh page", "error")
            return
        res = validate_image_file(profile_img_file)
        if "error" in res.keys():
            fh.add_toast(session, res["error"], "error")
            return (
                fh.Img(
                    src=curr_user.profile_img,
                    cls=f"size-12 object-cover rounded-full cursor-pointer hover:{img_hover}",
                    id="profile-img-preview",
                ),
            )
        return (
            fh.Img(
                src=f"data:image/png;base64,{res['success']}",
                cls=f"size-12 object-cover rounded-full cursor-pointer hover:{img_hover}",
                id="profile-img-preview",
            ),
        )

    @f_app.patch("/user/settings/save")
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
        if not curr_user:
            fh.add_toast(session, "Refresh page", "error")
            return None
        with get_db_session() as db_session:
            if email != curr_user.email:
                query = select(User).where(User.email == email)
                db_user = db_session.exec(query).first()
                if db_user:
                    fh.add_toast(session, "Email already exists", "error")
                    return None
                curr_user.email = email

            if username != curr_user.username:
                query = select(User).where(User.username == username)
                db_user = db_session.exec(query).first()
                if db_user:
                    fh.add_toast(session, "Username already exists", "error")
                    return None
                curr_user.username = username

            if curr_user.login_type == "email":
                if password:
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

    @f_app.delete("/user/settings/delete-account")
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

    @f_app.get("/cancel")
    def cancel():
        return fh.RedirectResponse("/")

    @f_app.get("/success")
    def success():
        return fh.RedirectResponse("/")

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
            with get_db_session() as db_session:
                curr_balance.balance += 5
                db_session.add(curr_balance)
                db_session.commit()
                db_session.refresh(curr_balance)
            return {"status": "success"}, 200

    ## misc
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
    image=FE_IMAGE,
    cpu=CPU,
    memory=MEM,
    secrets=SECRETS,
    timeout=5 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.asgi_app()
def modal_get():
    return f_app


if __name__ == "__main__":
    download_models()
    fh.serve(app="f_app")
