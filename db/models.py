import uuid
from datetime import datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

# global balance
init_balance = 100


class GlobalBalanceBase(SQLModel):
    balance: int = Field(default=init_balance, index=True)


class GlobalBalance(GlobalBalanceBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


class GlobalBalanceCreate(GlobalBalanceBase):
    pass


class GlobalBalanceRead(GlobalBalanceBase):
    id: int


class GlobalBalanceUpdate(SQLModel):
    balance: int | None = None


# user
init_user_balance = 100


class UserBase(SQLModel):
    balance: int = Field(default=init_user_balance)
    login_type: str | None = Field(default=None)  # github, google, email
    profile_img: str | None = Field(default=None)  # base64
    email: str | None = Field(default=None)
    username: str | None = Field(default=None)


class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default=str(uuid.uuid4()))

    hashed_password: str | None = Field(default=None)
    reset_token: str | None = Field(default=None)
    reset_token_expiry: datetime | None = Field(default=None)

    trials: list["Trial"] = Relationship(back_populates="user", cascade_delete=True)


class UserCreate(UserBase):
    password: str | None = Field(default=None)


class UserRead(UserBase):
    uuid: str


class UserUpdate(SQLModel):
    balance: int | None = None
    profile_img: str | None = None
    email: str | None = None
    password: str | None = None


# trial


class TrialBase(SQLModel):
    question: str
    success: bool | None = Field(default=None)


class Trial(TrialBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default=str(uuid.uuid4()))

    session_uuid: str | None = Field(default=None)

    user_id: int | None = Field(default=None, foreign_key="user.id", ondelete="CASCADE")
    user: User | None = Relationship(back_populates="trials")

    papers: list["Paper"] = Relationship(back_populates="trial", cascade_delete=True)


class TrialCreate(TrialBase):
    pass


class TrialRead(TrialBase):
    uuid: str
    session_uuid: str | None = None
    user: UserRead | None = None
    papers: list["PaperRead"] | None = None


class TrialUpdate(SQLModel):
    pass


# paper


class PaperBase(SQLModel):
    paper_id: str | None = Field(default=None, index=True)
    external_ids: dict | None = Field(default=None, sa_column=Column(JSON))
    title: str | None = Field(default=None, index=True)
    abstract: str | None = Field(default=None)
    venue: str | None = Field(default=None)
    year: int | None = Field(default=None)
    citation_count: int | None = Field(default=None)
    open_access_pdf: dict | None = Field(default=None, sa_column=Column(JSON))
    fields_of_study: list[str] | None = Field(default=None, sa_column=Column(JSON))
    publication_types: list[str] | None = Field(default=None, sa_column=Column(JSON))
    publication_date: str | None = Field(default=None)
    authors: list[dict] | None = Field(default=None, sa_column=Column(JSON))


class Paper(PaperBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default=str(uuid.uuid4()))

    trial_id: int | None = Field(
        default=None, foreign_key="trial.id", ondelete="CASCADE"
    )
    trial: Trial | None = Relationship(back_populates="papers")


class PaperCreate(PaperBase):
    pass


class PaperRead(PaperBase):
    uuid: str
    trial: TrialRead | None = None


class PaperUpdate(SQLModel):
    pass


# Update forward references to resolve circular dependencies
TrialRead.update_forward_refs()
PaperRead.update_forward_refs()
