"""Shared meaning of event-free BASE confidence and event-conditioned revision."""

BASE_INSTRUCTION = (
    'Generate BASE confidence for this instance under NO event. '
    'No rain or accident has occurred or been disclosed in this baseline. '
    'Do not anticipate a future event. BASE confidence is your own event-free '
    'assessment, not an observed performance measure or an optimality guarantee.'
)

EVENT_INSTRUCTION = (
    'The supplied prior_base_confidence contains the values YOU previously generated '
    'for this SAME instance and the SAME groups under NO event. '
    'Now identify the scope of the disclosed event: affected locations or customers, '
    'its time interval, and travel-time effects supported by the supplied information. '
    'Assess both direct exposure and possible downstream consequences for waiting, '
    'time-window slack, lateness and depot return, without inventing a route or arrival times. '
    'Revise those BASE values to reflect the event and the stated weighted objective. '
    'Being geographically affected does not automatically mean higher or lower confidence. '
    'Keep a BASE preference when the evidence does not justify changing it; '
    'make substantial changes when warranted. BASE values are revisable, not fixed targets. '
    'Keep group membership and the requested confidence output schema unchanged. '
    'Return the required structured confidence output, not a chain-of-thought narrative.'
)
