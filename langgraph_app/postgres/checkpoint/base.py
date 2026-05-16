from __future__ import annotations

import random
import warnings, uuid
from langchain_core.messages import message_to_dict, AIMessage, ToolMessage, HumanMessage, SystemMessage, BaseMessage, messages_from_dict
from collections.abc import Sequence
from importlib.metadata import version as get_version
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg.types.json import Jsonb

MetadataInput = dict[str, Any] | None

try:
    major, minor = get_version("langgraph").split(".")[:2]
    if int(major) == 0 and int(minor) < 5:
        warnings.warn(
            "You're using incompatible versions of langgraph and checkpoint-postgres. Please upgrade langgraph to avoid unexpected behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
except Exception:
    # skip version check if running from source
    pass

SELECT_SQL = """
select
    tenant_id,
    user_id,
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        from jsonb_each_text(checkpoint -> 'channel_versions')
        inner join "Session_blobs" bl
            on bl.thread_id = "Session_checkpoints".thread_id
            and bl.checkpoint_ns = "Session_checkpoints".checkpoint_ns
            and bl.channel = jsonb_each_text.key
            and bl.version = jsonb_each_text.value
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from "Session_checkpoint_writes" cw
        where cw.thread_id = "Session_checkpoints".thread_id
            and cw.checkpoint_ns = "Session_checkpoints".checkpoint_ns
            and cw.checkpoint_id = "Session_checkpoints".checkpoint_id
    ) as pending_writes
from "Session_checkpoints" """

SELECT_PENDING_SENDS_SQL = f"""
select
    tenant_id,
    user_id,
    checkpoint_id,
    array_agg(array[type::bytea, blob] order by task_path, task_id, idx) as sends
from "Session_checkpoint_writes"
where thread_id = %s
    and checkpoint_id = any(%s)
    and channel = '{TASKS}'
group by checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT INTO "Session_blobs" (tenant_id, user_id, thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO "Session_checkpoints" (tenant_id, user_id, thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO "Session_checkpoint_writes" (tenant_id, user_id, thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO "Session_checkpoint_writes" (tenant_id, user_id, thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"""

def convert_message_to_dict(obj):
    if isinstance(obj, BaseMessage):
        return message_to_dict(obj)
    elif isinstance(obj, list):
        return [convert_message_to_dict(i) for i in obj]
    
    return obj

def restore_uuid_keys(obj):
    if isinstance(obj, dict):
        restored = {}
        for k, v in obj.items():
            try:
                restored[uuid.UUID(k)] = restore_uuid_keys(v)
            except (ValueError, AttributeError):
                restored[k] = restore_uuid_keys(v)
        return restored
    elif isinstance(obj, list):
        return [restore_uuid_keys(i) for i in obj]
    return obj

class BasePostgresSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    supports_pipeline: bool

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[bytes, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends],
        )
        channel_values.append((TASKS.encode(), enc.encode(), blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        # print("masuk _load_blobs")
        if not blob_values:
            return {}
            
        result = {}
        for k, t, v in blob_values:
            key = k.decode()
            type_ = t.decode()
            
            if type_ == "empty":
                continue
            
            data_k = self.serde.loads_typed((type_, v))
            
            if key == "messages":
                if isinstance(data_k, dict):
                    try:
                        data_k = messages_from_dict([data_k])[0]
                    except Exception:
                        pass
                elif isinstance(data_k, list):
                    try:
                        data_k = data_k = messages_from_dict(data_k)
                    except Exception:
                        pass
                    
            result[key] = data_k
            
        return result  #{
            # k.decode(): self.serde.loads_typed((t.decode(), v))
            # for k, t, v in blob_values
            # if t.decode() != "empty"
        # }

    def _dump_blobs(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        # print("masuk _dump_blobs")
        if not versions:
            return []
            
        result = []
        for k, ver in versions.items():
            if k in values:
                data_k = convert_message_to_dict(values[k])
                type_, blob = self.serde.dumps_typed(
                    data_k #convert_message_to_dict(values[k])
                )
            else:
                type_, blob = ("empty", None)
                
            result.append(
                (
                    tenant_id,
                    user_id,
                    thread_id,
                    checkpoint_ns,
                    k,
                    cast(str, ver),
                    type_,
                    blob,
                )
            )
                
        
        return result #[
            # (
                # tenant_id,
                # user_id,
                # thread_id,
                # checkpoint_ns,
                # k,
                # cast(str, ver),
                # *(
                    # self.serde.dumps_typed(convert_message_to_dict(values[k]))
                    # if k in values
                    # else ("empty", None)
                # ),
            # )
            # for k, ver in versions.items()
        # ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        # print("masuk _load_writes")
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        # print("masuk _dump_writes")
        return [
            (
                tenant_id,
                user_id,
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )