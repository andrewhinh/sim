from sqlmodel import Field, SQLModel

### global balance
init_balance = 100


class GlobalBalanceBase(SQLModel):
    balance: int = Field(default=init_balance, index=True)


class GlobalBalance(GlobalBalanceBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


class GlobalBalanceCreate(GlobalBalanceBase):
    pass


class GlobalBalanceRead(GlobalBalanceBase):
    id: int
