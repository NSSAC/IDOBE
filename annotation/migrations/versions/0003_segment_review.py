"""segment review

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('series',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('series_key', sa.String(length=255), nullable=False),
    sa.Column('source', sa.String(length=128), nullable=True),
    sa.Column('disease', sa.String(length=128), nullable=False),
    sa.Column('location', sa.String(length=128), nullable=False),
    sa.Column('geo_level', sa.String(length=32), nullable=True),
    sa.Column('event', sa.String(length=64), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('values', sa.JSON(), nullable=False),
    sa.Column('loaded_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_series_series_key', 'series', ['series_key'], unique=True)
    op.create_index('ix_series_disease', 'series', ['disease'], unique=False)
    op.create_table('segmentation_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('method', sa.String(length=128), nullable=False),
    sa.Column('method_version', sa.String(length=64), nullable=True),
    sa.Column('source_file', sa.String(length=255), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('run_series',
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('series_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['segmentation_runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['series_id'], ['series.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('run_id', 'series_id')
    )
    op.create_table('proposed_segments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('series_id', sa.Integer(), nullable=False),
    sa.Column('segment_key', sa.String(length=255), nullable=False),
    sa.Column('start_idx', sa.Integer(), nullable=False),
    sa.Column('end_idx', sa.Integer(), nullable=False),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('peak_idx', sa.Integer(), nullable=True),
    sa.Column('core_start_idx', sa.Integer(), nullable=True),
    sa.Column('core_end_idx', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['segmentation_runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['series_id'], ['series.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_proposed_segments_run_series', 'proposed_segments', ['run_id', 'series_id'], unique=False)
    op.create_table('segment_reviews',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('series_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('items', sa.JSON(), nullable=False),
    sa.Column('series_unusable', sa.Boolean(), nullable=False),
    sa.Column('skipped', sa.Boolean(), nullable=False),
    sa.Column('confidence', sa.Integer(), nullable=True),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('time_spent_ms', sa.Integer(), nullable=True),
    sa.Column('app_version', sa.String(length=32), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('confidence IS NULL OR confidence BETWEEN 1 AND 5', name='ck_segment_review_confidence'),
    sa.ForeignKeyConstraint(['run_id'], ['segmentation_runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['series_id'], ['series.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_segment_reviews_run_series_user', 'segment_reviews',
                    ['run_id', 'series_id', 'user_id', 'id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_segment_reviews_run_series_user', table_name='segment_reviews')
    op.drop_table('segment_reviews')
    op.drop_index('ix_proposed_segments_run_series', table_name='proposed_segments')
    op.drop_table('proposed_segments')
    op.drop_table('run_series')
    op.drop_table('segmentation_runs')
    op.drop_index('ix_series_disease', table_name='series')
    op.drop_index('ix_series_series_key', table_name='series')
    op.drop_table('series')
