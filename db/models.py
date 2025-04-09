import uuid
from datetime import datetime
from sqlmodel import Field, SQLModel

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


# user
init_user_balance = 100


class UserBase(SQLModel):
    balance: int = Field(default=init_user_balance)
    login_type: str | None = Field()  # github, google, email
    profile_img: str | None = Field()  # base64
    email: str | None = Field()
    username: str | None = Field()


class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default=str(uuid.uuid4()))
    hashed_password: str | None = Field()
    reset_token: str | None = Field(default=None)
    reset_token_expiry: datetime | None = Field(default=None)


class UserCreate(UserBase):
    password: str | None = Field()


class UserRead(UserBase):
    uuid: str


class UserUpdate(SQLModel):
    balance: int | None = None
    profile_img: str | None = None
    email: str | None = None
    password: str | None = None
