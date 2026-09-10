package taxonomy

// RoutePolicy is the auto/escalate annotation guideline. The prompts embed it
// verbatim and the annotation guide is generated from it.
const RoutePolicy = `AUTO vs ESCALATE - annotation rule

Label a message "auto" ONLY IF ALL of these hold:
  1. A short, public, generic reply fully serves this customer. Nothing further is owed.
  2. The reply needs no account, booking or payment data, and no lookup of this
     customer's specific situation.
  3. No money, miles or vouchers are at stake.
  4. Nothing in the message raises safety, medical, legal, discrimination or
     accessibility concerns.
  5. The customer is not angry enough that a templated reply would make things worse.
  6. Delta is not exposed reputationally (press, virality, a public thread with an
     audience).

Otherwise label it "escalate".

Notes for edge cases:
  - Pure thanks/compliments are "auto" - a warm acknowledgement is the whole job.
  - "How do I reach a human / I've been on hold" is "auto": the correct reply is to
    route them to the right channel, which needs no account access.
  - Generic policy questions (baggage allowance, pet policy) are "auto" IF the answer
    does not depend on the customer's specific booking.
  - Anything where Delta's real reply asked the customer to DM is almost always
    "escalate" - the DM request is Delta itself saying a human must take over.
  - A message that is mostly praise but ends with a real request is "escalate", and its
    intent is the request, not praise.
  - Judge by what the message NEEDS, not by how Delta happened to answer it. Delta's
    actual reply is evidence, not the label.`
