# sim

Directed interventional in-silico simulations

![Website diagram](./sim.excalidraw.png)

## Development

### Set Up

Set up the environment:

```bash
make setup
```

Create a `.env` (+ `.env.dev`):

```bash
HF_TOKEN=

POSTGRES_URL=
POSTGRES_PRISMA_URL=
SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_URL=
POSTGRES_URL_NON_POOLING=
SUPABASE_JWT_SECRET=
POSTGRES_USER=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
POSTGRES_PASSWORD=
POSTGRES_DATABASE=
SUPABASE_SERVICE_ROLE_KEY=
POSTGRES_HOST=
SUPABASE_ANON_KEY=

GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=


STRIPE_PUBLISHABLE_KEY=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=

EMAIL_SENDER=
SMTP_SERVER=
SMTP_PORT=
SMTP_USERNAME=
SMTP_PASSWORD=

DOMAIN=
```

### Useful Tips

Migrate db (do before running the frontend/api):

```bash
make migrate MSG="your migration message" ENV=dev
```

### Repository Structure

```bash
.
├── assets              # assets.
├── db                  # database.
├── src                 # frontend.
```

### Frontend

Serve the web app locally:

```bash
uv run src/app.py
stripe listen --forward-to <url>/webhook
# update STRIPE_WEBHOOK_SECRET and DOMAIN in .env.dev
```

Serve the web app on Modal:

```bash
modal serve src/app.py
stripe listen --forward-to <url>/webhook
# update STRIPE_WEBHOOK_SECRET and DOMAIN in .env.dev
```

Deploy on dev:

```bash
modal deploy src/app.py
# update STRIPE_WEBHOOK_SECRET and DOMAIN in .env.dev
```

Deploy on main:

```bash
modal deploy --env=main src/app.py
```
