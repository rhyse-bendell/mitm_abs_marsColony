"""Persistent record of DIK requests an agent has RECEIVED (the requester's information needs, not the recipient's own gaps).

Lifecycle of one request, kept distinct on purpose:

    received -> response selected (the recipient's Pilot chose to respond)
             -> delivery attempted (a communication action reached the requester, or failed on range)
             -> provided (elements the recipient actually holds were transmitted)
             -> uptake (the requester's mental model changed; measured, not known to the recipient)
             -> requester gap resolved (the requester's own bookkeeping)

This ledger owns the recipient-side part: what was asked, what the recipient can answer now, what it has provided, and when
the request is closed. It reads nothing about the requester except what the request message said. Closing reasons are
``provided_all`` (every requested element was transmitted), ``expired`` (no longer actionable after ``REQUEST_TTL_S``) and
``withdrawn`` (the requester's newer request no longer asks for it). A provision never marks a request "satisfied": whether the
requester acquired the element is the requester's own state.
"""

# Hypothesized simulator default (not an empirical estimate): how long a received request stays actionable.
REQUEST_TTL_S = 120.0


class ReceivedRequestLedger:
    def __init__(self, ttl_s=REQUEST_TTL_S):
        self.ttl_s = float(ttl_s)
        self.active = {}  # requester name -> record
        self.history = []  # closed records, oldest first
        self._counter = 0

    # ---- intake
    def receive(self, requester, requested_ids, now, project_id=None):
        """Register a request message; returns (record, is_new). A newer request from the same requester replaces the asked set."""
        requested = {str(i) for i in requested_ids}
        record = self.active.get(requester)
        if record is not None and not requested:
            self._close(record, now, "withdrawn")
            return record, False
        if not requested:
            return None, False
        if record is None:
            self._counter += 1
            record = {
                "request_id": f"req{self._counter}:{requester}@{round(float(now), 1)}",
                "requester": requester,
                "requested_ids": set(requested),
                "requested_at": float(now),
                "last_updated_at": float(now),
                "delivered": {},
                "withdrawn_ids": set(),
                "project_id": project_id,
                "selections": 0,
                "delivery_attempts": 0,
                "closed_at": None,
                "closed_reason": None,
            }
            self.active[requester] = record
            return record, True
        dropped = record["requested_ids"] - requested
        record["withdrawn_ids"] |= dropped
        record["requested_ids"] = set(requested)
        record["last_updated_at"] = float(now)
        # Re-asking for something already provided means the requester still lacks it; a duplicate that predates the
        # delivery (same or earlier time) is not a new need, so no second reply is owed.
        for eid in list(record["delivered"]):
            if eid in requested and float(now) > record["delivered"][eid]:
                record["delivered"].pop(eid)
            elif eid not in requested:
                record["delivered"].pop(eid)
        return record, False

    # ---- state
    def expire(self, now):
        """Close requests older than the TTL (measured from the latest update); returns the closed records."""
        closed = []
        for requester, record in list(self.active.items()):
            if float(now) - record["last_updated_at"] > self.ttl_s:
                self._close(record, now, "expired")
                closed.append(record)
        return closed

    def open_record(self, requester):
        return self.active.get(requester)

    def open_records(self):
        return [self.active[k] for k in sorted(self.active)]

    @staticmethod
    def outstanding_ids(record):
        return set(record["requested_ids"]) - set(record["delivered"])

    def actionable(self, requester, held_tokens):
        """(ids the recipient holds and has not yet provided, ids still asked for that it does not hold)."""
        record = self.active.get(requester)
        if record is None:
            return [], []
        outstanding = self.outstanding_ids(record)
        have = sorted(outstanding & set(held_tokens))
        lack = sorted(outstanding - set(held_tokens))
        return have, lack

    def mark_selected(self, requester):
        record = self.active.get(requester)
        if record is not None:
            record["selections"] += 1
        return record

    def mark_attempt(self, requester):
        record = self.active.get(requester)
        if record is not None:
            record["delivery_attempts"] += 1
        return record

    def mark_provided(self, requester, provided_ids, now):
        """Record elements actually transmitted. Closes the request when nothing asked for remains undelivered."""
        record = self.active.get(requester)
        if record is None:
            return None
        for eid in provided_ids:
            if eid in record["requested_ids"]:
                record["delivered"][eid] = float(now)
        if not self.outstanding_ids(record):
            self._close(record, now, "provided_all")
        return record

    def status(self, record):
        if record["closed_reason"]:
            return record["closed_reason"]
        return "partially_provided" if record["delivered"] else "open"

    def _close(self, record, now, reason):
        record["closed_at"] = float(now)
        record["closed_reason"] = reason
        if self.active.get(record["requester"]) is record:
            del self.active[record["requester"]]
        self.history.append(record)
