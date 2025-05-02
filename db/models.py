from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

# global balance
init_balance = 100


class GlobalBalanceBase(SQLModel):
    balance: int = Field(default=init_balance, index=True)


class GlobalBalance(GlobalBalanceBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


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
    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hashed_password: str | None = Field(default=None)
    reset_token: str | None = Field(default=None)
    reset_token_expiry: datetime | None = Field(default=None)

    trials: list["Trial"] = Relationship(back_populates="user", cascade_delete=True)


class UserCreate(UserBase):
    password: str | None = Field(default=None)


class UserRead(UserBase):
    uuid: str
    trials: list["Trial"] | None = None


# trial


class TrialBase(SQLModel):
    question: str

    data_point_extract_complete: bool | None = Field(default=None)
    rerank_and_visualize_complete: bool | None = Field(default=None)
    success: bool | None = Field(default=None)


class Trial(TrialBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid4()))

    session_uuid: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    user_id: int | None = Field(default=None, foreign_key="user.id", ondelete="CASCADE")
    user: User | None = Relationship(back_populates="trials")

    search_params: Optional["SearchParams"] = Relationship(
        back_populates="trial", cascade_delete=True
    )
    papers: list["Paper"] | None = Relationship(
        back_populates="trial",
        cascade_delete=True,
    )
    visualization: dict | None = Field(default=None, sa_column=Column(JSON))


# search params


class SearchParamsBase(SQLModel):
    query: str | None = None


class SearchParams(SearchParamsBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid4()))

    trial_id: int | None = Field(
        default=None, foreign_key="trial.id", ondelete="CASCADE"
    )
    trial: Trial | None = Relationship(back_populates="search_params")


# paper


class PaperBase(SQLModel):
    paper_id: str | None = Field(default=None, index=True)
    corpus_id: int | None = Field(default=None, index=True)
    external_ids: dict | None = Field(default=None, sa_column=Column(JSON))
    url: str | None = Field(default=None)
    title: str | None = Field(default=None, index=True)
    abstract: str | None = Field(default=None)
    venue: str | None = Field(default=None)
    publication_venue: dict | None = Field(default=None, sa_column=Column(JSON))
    year: int | None = Field(default=None)
    reference_count: int | None = Field(default=None)
    citation_count: int | None = Field(default=None)
    influential_citation_count: int | None = Field(default=None)
    is_open_access: bool | None = Field(default=None)
    open_access_pdf: dict | None = Field(default=None, sa_column=Column(JSON))
    fields_of_study: list[str] | None = Field(default=None, sa_column=Column(JSON))
    s2_fields_of_study: list[dict] | None = Field(default=None, sa_column=Column(JSON))
    publication_types: list[str] | None = Field(default=None, sa_column=Column(JSON))
    publication_date: str | None = Field(default=None)
    journal: dict | None = Field(default=None, sa_column=Column(JSON))
    citation_styles: dict | None = Field(default=None, sa_column=Column(JSON))
    authors: list[dict] | None = Field(default=None, sa_column=Column(JSON))

    chunks: list[str] | None = Field(default=None, sa_column=Column(JSON))


class Paper(PaperBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid4()))

    trial_id: int | None = Field(
        default=None, foreign_key="trial.id", ondelete="CASCADE"
    )
    trial: Trial | None = Relationship(back_populates="papers")

    data_points: list["DataPoint"] | None = Relationship(
        back_populates="paper", cascade_delete=True
    )

    def __str__(self):
        paper = self.dict()
        return f"""
            Title: {paper.get("title", "")}
            Abstract: {paper.get("abstract", "")}
            Venue: {paper.get("venue", "")}
            Publication Venue: {paper.get("publication_venue", {})}
            Year: {paper.get("year", "")}
            Reference Count: {paper.get("reference_count", "")}
            Citation Count: {paper.get("citation_count", "")}
            Influential Citation Count: {paper.get("influential_citation_count", "")}
            Fields of Study: {paper.get("fields_of_study", "")}
            Publication Types: {paper.get("publication_types", "")}
            Publication Date: {paper.get("publication_date", "")}
            Journal: {paper.get("journal", "")}
            Authors: {"\n\n".join(
                [
                    f"""
                    Name: {author.get("name", "")}
                    Affiliations: {', '.join(author.get("affiliations", []))}
                    Paper Count: {author.get("paper_count", "")}
                    Citation Count: {author.get("citation_count", "")}
                    H-Index: {author.get("h_index", "")}
                    """
                    for author in paper.get("authors", [])
                ]
            )}
            Text: {"\n\n".join(paper.get("chunks", []))}
            Extracted data points: {"\n\n".join([str(dp) for dp in paper.get("data_points", [])])}
        """


# data points


class DataPointBase(SQLModel):
    name: str
    value: float
    unit: str | None = None
    excerpt: str | None = None

    def __str__(self):
        return f"{self.name}: {self.value} {self.unit} ({self.excerpt})"


class DataPoint(DataPointBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid4()))

    paper_id: int | None = Field(foreign_key="paper.id", ondelete="CASCADE")
    paper: Paper | None = Relationship(back_populates="data_points")
