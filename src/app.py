import os
import uuid
from asyncio import sleep
from contextlib import contextmanager

import modal
import stripe
from fasthtml import common as fh
from simpleicons.icons import si_github
from sqlmodel import Session as DBSession
from sqlmodel import create_engine
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from db.models import (
    GlobalBalance,
    GlobalBalanceCreate,
    GlobalBalanceRead,
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
    text_color = "stone-700"
    background_color = "stone-100"
    font = "font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'"

    img_hover = "hover:opacity-75"
    text_hover_color = "stone-500"
    click_color = "blue-400"
    click_hover_color = "blue-500"
    error_color = "red-400"
    input_color = "stone-50"

    main_page = f"flex flex-col justify-between min-h-screen w-full bg-{background_color} text-{text_color} {font}"
    page_ctnt = "flex flex-col justify-center items-center grow gap-4 p-8"
    button = f"text-{text_color} bg-{click_color} hover:bg-{click_hover_color} cursor-pointer rounded-full px-3 py-1"

    # setup
    def before(req, sess):
        if "session_id" not in sess:
            req.scope["session_id"] = sess.setdefault("session_id", str(uuid.uuid4()))
        if "user_id" not in sess:
            sess["user_id"] = ""  # set when user creates account or logs in
        if "login_type" not in sess:
            sess["login_type"] = ""  # set when user creates account or logs in

    def _not_found(req, exc):
        message = "Page not found!"
        return (
            fh.Title(APP_NAME + " | 404"),
            fh.Div(
                nav(),
                fh.Main(
                    fh.P(
                        message,
                        hx_indicator="#spinner",
                        cls=f"text-2xl text-{error_color} overflow-hidden whitespace-nowrap",
                    ),
                    cls=page_ctnt,
                ),
                toast_container(),
                footer(),
                cls=main_page,
            ),
        )

    f_app, _ = fh.fast_app(
        ws_hdr=True,
        before=fh.Beforeware(
            before, skip=[r"/favicon\.ico", r"/static/.*", r".*\.css"]
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

    ## db
    engine = create_engine(url=os.getenv("POSTGRES_URL"), echo=False)

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

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

    ## stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    webhook_secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    DOMAIN = os.environ["DOMAIN"]

    ## SSE state
    shutdown_event = fh.signal_shutdown()
    global shown_balance
    shown_balance = 0

    ## layout
    def nav():
        return fh.Nav(
            fh.Div(
                fh.A(
                    fh.Img(
                        src="/logo.png",
                        cls="w-10 h-10",
                    ),
                    fh.P(
                        APP_NAME,
                        cls=f"text-lg text-{text_color}",
                    ),
                    href="/",
                    cls=f"flex items-center gap-2 {img_hover}",
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
                cls="htmx-indicator w-8 h-8 absolute top-8 left-1/2 transform -translate-x-1/2 fill-blue-300",
            ),
            fh.Div(
                fh.A(
                    fh.P(
                        "Log In",
                        cls=f"text-{text_color} hover:text-{text_hover_color}",
                    ),
                    href="/login",
                ),
                fh.A(
                    fh.P("Sign Up", cls=button),
                    href="/signup",
                ),
                cls="flex flex-col items-end md:flex-row md:items-center gap-2 md:gap-4",
            ),
            cls="flex justify-between items-center p-4 relative",
        )

    def main_content():
        return fh.Main(
            fh.Form(
                fh.Textarea(
                    id="user-prompt",
                    placeholder="Ask a research question...",
                    rows=10,
                    cls=f"bg-{input_color} p-3",
                    onkeydown="if(event.key === 'Enter' && event.shiftKey) { this.form.submit(); event.preventDefault(); }",
                ),
                fh.Button(
                    fh.P(
                        "→",
                        cls=f"absolute bottom-0.5 right-2 text-{text_color} hover:text-{text_hover_color}",
                    ),
                    type="submit",
                    cls=f"{button} absolute bottom-0 right-2",
                    style="width: 2rem; height: 2rem;",
                ),
                cls="relative w-full md:w-2/3 flex justify-between items-center",
            ),
            fh.P(
                "Or try some examples:",
                cls=f"text-{text_color}",
            ),
            fh.Div(
                *[
                    fh.Button(
                        prompt,
                        cls=button,
                        hx_post="/toast",
                    )
                    for prompt in DEFAULT_USER_PROMPTS
                ],
                cls="flex flex-col md:flex-row gap-4",
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
                    cls=f"flex items-start gap-0.5 md:gap-1 text-{text_color}",
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
                cls="flex flex-col gap-0.5",
            ),
            fh.Div(
                fh.A(
                    fh.Svg(
                        fh.NotStr(
                            si_github.svg,
                        ),
                        cls=f"w-8 h-8 hover:text-{text_hover_color}",
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
                    cls="flex flex-col text-right gap-0.5",
                ),
                cls="flex flex-col md:flex-row items-end md:items-center gap-2 md:gap-4",
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

    # routes
    ## for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    def static_files(fname: str, ext: str):
        static_file_path = PARENT_PATH / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ## toasts without target
    @f_app.post("/toast")
    def toast(session, message: str, type: str):
        fh.add_toast(session, message, type)
        return toast_container()

    ## pages
    @f_app.get("/")
    def home(
        session,
    ):
        return (
            fh.Title(APP_NAME),
            fh.Div(
                nav(),
                main_content(),
                toast_container(),
                footer(),
                cls=main_page,
            ),
        )

    @f_app.get("/stream-balance")
    async def stream_balance():
        """Stream balance updates to connected clients"""
        return StreamingResponse(
            stream_balance_updates(), media_type="text/event-stream"
        )

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
