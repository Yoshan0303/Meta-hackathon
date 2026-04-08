"""
Synthetic email dataset generator for EmailTriageEnv.

Generates 200 realistic emails across 5 categories with deterministic
ground-truth labels. Uses seed=42 for full reproducibility.

Context: A mid-size B2B SaaS company ("CloudMetrics") that provides
analytics and monitoring tools. Emails come from customers, partners,
internal teams, and spammers.
"""

import random
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Ground-truth label type
# ---------------------------------------------------------------------------

PRIORITY_LEVELS = ("urgent", "high", "medium", "low")
CATEGORIES = ("bug_report", "feature_request", "billing", "support", "spam")
ACTIONS = ("reply", "escalate", "archive", "delete", "snooze")


@dataclass
class GroundTruthEmail:
    """A single email with baked-in ground-truth labels (hidden from agent)."""

    email_id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    timestamp: str
    thread_length: int
    has_attachment: bool
    # Ground-truth labels — never exposed to the agent
    true_priority: str
    true_category: str
    expected_action: str
    reply_keywords: list[str] = field(default_factory=list)
    is_adversarial: bool = False


# ---------------------------------------------------------------------------
# Sender pools
# ---------------------------------------------------------------------------

_LEGIT_SENDERS = [
    ("James Chen", "acmecorp.com"),
    ("Sarah Mitchell", "globex.io"),
    ("Raj Patel", "initech.com"),
    ("Emily Zhao", "umbrellacorp.net"),
    ("Marcus Johnson", "wayneenterprises.com"),
    ("Lisa Nakamura", "starkindustries.co"),
    ("David Kim", "oscorp.com"),
    ("Ana Rodriguez", "lexcorp.biz"),
    ("Tom Bradley", "cyberdyne.systems"),
    ("Priya Sharma", "soylent.io"),
    ("Michael O'Brien", "bigbank.com"),
    ("Fatima Al-Rashid", "petromax.ae"),
    ("Carlos Mendez", "latamlogistics.co"),
    ("Wei Zhang", "shenzhentech.cn"),
    ("Ingrid Larsson", "nordicsaas.se"),
    ("Yuki Tanaka", "tokyodata.jp"),
    ("Olga Kuznetsova", "rucloud.ru"),
    ("Jean-Pierre Dubois", "eurotrade.fr"),
    ("Amara Obi", "lagosfintech.ng"),
    ("Sofia Costa", "lisboatech.pt"),
]

_INTERNAL_SENDERS = [
    ("Alex Rivera", "cloudmetrics.com"),
    ("Jordan Hayes", "cloudmetrics.com"),
    ("Sam Nguyen", "cloudmetrics.com"),
    ("Casey Morgan", "cloudmetrics.com"),
    ("Taylor Brooks", "cloudmetrics.com"),
]

_SPAM_SENDERS = [
    ("Prize Notification", "fr33-m0ney.xyz"),
    ("Account Security", "securealert-update.click"),
    ("Dr. Robert", "pharmadeals99.biz"),
    ("LinkedIn", "linkedln-verify.info"),
    ("Apple Support", "apple-id-secure.top"),
    ("Crypto Signals", "10xgains-crypto.io"),
    ("HR Department", "company-benefits-update.net"),
    ("Webinar Invite", "exclusive-masterclass.guru"),
]

# ---------------------------------------------------------------------------
# Email templates per category
# ---------------------------------------------------------------------------


def _bug_reports(rng: random.Random) -> list[dict]:
    """Generate bug-report emails of varying severity."""
    templates = [
        # URGENT bugs
        {
            "subject": "CRITICAL: Dashboard showing zero data for all customers",
            "body": (
                "Hi CloudMetrics team,\n\n"
                "Since 3:00 AM UTC this morning, our entire dashboard is blank. "
                "All charts show zero values and the data pipeline status shows 'stalled'. "
                "This is affecting all 2,500+ of our end users. We have an investor demo "
                "in 4 hours and NEED this resolved immediately.\n\n"
                "Error code we're seeing: ERR_PIPELINE_TIMEOUT_5003\n\n"
                "Please escalate to your on-call engineering team ASAP.\n\n"
                "—James"
            ),
            "priority": "urgent",
            "action": "escalate",
            "reply_keywords": ["investigating", "pipeline", "priority", "engineering"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Production API returning 500 errors — blocking our release",
            "body": (
                "Team,\n\n"
                "Our integration with the CloudMetrics API has been returning 500 errors "
                "for the past 45 minutes. All POST requests to /v2/events/ingest are failing. "
                "We've verified it's not on our end — other API providers are working fine.\n\n"
                "This is blocking our production deployment scheduled for today. "
                "We're an Enterprise tier customer (Account #ENT-4892).\n\n"
                "HTTP response:\n"
                "```\n{\"error\": \"internal_server_error\", \"request_id\": \"req_8f3a2b\"}\n```\n\n"
                "Please investigate urgently.\n\nBest,\nSarah Mitchell"
            ),
            "priority": "urgent",
            "action": "escalate",
            "reply_keywords": ["API", "500", "investigating", "engineering", "status"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # HIGH bugs
        {
            "subject": "Export to CSV generating corrupted files",
            "body": (
                "Hi there,\n\n"
                "When I export reports to CSV from the Analytics tab, the downloaded file "
                "has garbled characters after row 10,000. This started happening after your "
                "last update (v4.2.1). It works fine for smaller datasets.\n\n"
                "I've attached a screenshot of the corrupted output. This is impacting our "
                "monthly reporting workflow — we need CSVs for our finance team by Friday.\n\n"
                "Browser: Chrome 120.0\nOS: macOS Sonoma\n\nThanks,\nRaj"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["CSV", "export", "fix", "investigating", "update", "workaround"],
            "thread_length": 1,
            "has_attachment": True,
        },
        {
            "subject": "SSO login broken after your maintenance window",
            "body": (
                "Hello,\n\n"
                "After your scheduled maintenance last night, our SAML SSO integration is "
                "no longer working. Users get redirected to a 404 page after authenticating "
                "with our IdP (Okta). This is affecting ~150 users in our org.\n\n"
                "We've confirmed our Okta configuration hasn't changed. I suspect something "
                "on your callback URL handling broke during the update.\n\n"
                "Can you check the SAML assertion consumer URL for our tenant?\n\n"
                "Regards,\nEmily Zhao\nIT Director, Umbrella Corp"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["SSO", "SAML", "investigating", "configuration", "fix"],
            "thread_length": 2,
            "has_attachment": False,
        },
        # MEDIUM bugs
        {
            "subject": "Chart tooltips overlapping on mobile view",
            "body": (
                "Hey,\n\n"
                "Noticed that when viewing dashboards on my iPad, the chart tooltips "
                "overlap with the legend. It's not a blocker but a bit annoying. "
                "Seems like a CSS z-index issue maybe?\n\n"
                "Happens in both Safari and Chrome on iPad.\n\nCheers,\nMarcus"
            ),
            "priority": "medium",
            "action": "archive",
            "reply_keywords": ["mobile", "tooltip", "UI", "fix", "backlog"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Date picker shows wrong timezone for AU users",
            "body": (
                "Hi support,\n\n"
                "The date range picker in the reports section seems to default to UTC "
                "instead of our local timezone (AEST). We set our org timezone to "
                "Australia/Sydney in settings, but the picker still shows UTC dates.\n\n"
                "It's causing a bit of confusion when filtering — off by one day issues.\n\n"
                "Not super urgent but would be nice to fix.\n\nThanks,\nLisa"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["timezone", "date", "fix", "settings", "known issue"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Webhook retry logic not working as documented",
            "body": (
                "Hello,\n\n"
                "According to your docs, failed webhook deliveries should retry 3 times "
                "with exponential backoff. We're only seeing a single retry attempt in our "
                "logs. We've tested with an endpoint that returns 503 intentionally.\n\n"
                "Not critical for us right now since we have our own retry logic, but "
                "thought you should know the behavior doesn't match the docs.\n\n"
                "Best,\nDavid Kim"
            ),
            "priority": "medium",
            "action": "archive",
            "reply_keywords": ["webhook", "retry", "documentation", "investigating"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # LOW bugs
        {
            "subject": "Typo in the 'Organizaiton Settings' page header",
            "body": (
                "Just noticed there's a typo in the settings page. The header says "
                "'Organizaiton Settings' instead of 'Organization Settings'. Small thing!\n\n"
                "—Ana"
            ),
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["typo", "fix", "thanks"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Favicon not showing in Firefox",
            "body": (
                "Hi there, minor cosmetic issue — the CloudMetrics favicon doesn't "
                "appear in the Firefox tab. Works fine in Chrome and Edge though. "
                "Running Firefox 121 on Windows 11.\n\nJust FYI!\n\nTom"
            ),
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["favicon", "Firefox", "noted"],
            "thread_length": 1,
            "has_attachment": False,
        },
    ]
    return templates


def _feature_requests(rng: random.Random) -> list[dict]:
    """Generate feature request emails."""
    templates = [
        # HIGH
        {
            "subject": "Request: Custom role-based access control (RBAC)",
            "body": (
                "Hi CloudMetrics team,\n\n"
                "We're onboarding 50 new users next month and desperately need granular "
                "role-based access control. Right now we only have Admin and Viewer roles, "
                "but we need:\n\n"
                "- Dashboard Editor (can edit dashboards but not settings)\n"
                "- Data Analyst (read-only + export permissions)\n"
                "- Billing Manager (only billing section access)\n\n"
                "This is actually becoming a blocker for our enterprise expansion. We'd "
                "potentially upgrade to your Enterprise tier if RBAC is available.\n\n"
                "Is this on your roadmap?\n\nBest,\nPriya Sharma\nHead of Engineering, Soylent.io"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["RBAC", "roles", "roadmap", "enterprise", "feature"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Need: API rate limit increase for our data pipeline",
            "body": (
                "Hello,\n\n"
                "We're hitting the 1000 req/min API rate limit frequently now. Our data "
                "pipeline processes events from 50+ microservices and we're growing fast. "
                "Can we get a limit increase to 5000 req/min?\n\n"
                "We're on the Growth plan but happy to discuss upgrading if needed.\n\n"
                "Thanks,\nWei Zhang"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["rate limit", "API", "upgrade", "plan", "increase"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # MEDIUM
        {
            "subject": "Suggestion: Dark mode for the dashboard",
            "body": (
                "Would love to see a dark mode option for the CloudMetrics dashboard. "
                "Our team works late hours monitoring and the bright white theme is harsh "
                "on the eyes.\n\n"
                "A simple toggle in the user settings would be great.\n\n"
                "Thanks!\nIngrid Larsson"
            ),
            "priority": "medium",
            "action": "archive",
            "reply_keywords": ["dark mode", "UI", "roadmap", "noted"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Feature: Slack integration for alert notifications",
            "body": (
                "Hi,\n\n"
                "We currently get alert emails but our team operates mainly through Slack. "
                "Would be fantastic to have a native Slack integration that posts alerts "
                "to designated channels with rich formatting.\n\n"
                "I saw a competitor (DataDog) has this and it's really useful. Any plans "
                "to add Slack support?\n\n"
                "Cheers,\nYuki Tanaka"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["Slack", "integration", "roadmap", "alerts", "notifications"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Can we get PDF export for reports?",
            "body": (
                "The CSV export is great, but our executives prefer PDF reports they can "
                "share in board meetings. Would be nice to have a 'Export as PDF' button "
                "that generates a formatted report with charts included.\n\n"
                "Low urgency but would save us a lot of manual screenshot work.\n\n"
                "Thanks,\nMichael O'Brien"
            ),
            "priority": "medium",
            "action": "archive",
            "reply_keywords": ["PDF", "export", "report", "feature"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # LOW
        {
            "subject": "Idea: Customizable dashboard color themes",
            "body": (
                "Just a small suggestion — it would be cool if we could customize the "
                "chart color palette to match our brand colors. Not a big deal but would "
                "make embedded dashboards look more professional.\n\n"
                "Love the product otherwise!\n\nSofia"
            ),
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["colors", "theme", "customization", "noted"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Would be nice: Remember my last dashboard tab",
            "body": (
                "Minor UX thing — every time I log in, the dashboard defaults to the "
                "Overview tab. I always go to the 'Real-time' tab. Would be convenient "
                "if it remembered my last viewed tab.\n\n"
                "Thanks,\nJean-Pierre"
            ),
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["UX", "tab", "remember", "convenience"],
            "thread_length": 1,
            "has_attachment": False,
        },
    ]
    return templates


def _billing_emails(rng: random.Random) -> list[dict]:
    """Generate billing-related emails."""
    templates = [
        # URGENT
        {
            "subject": "URGENT: Double charged on our last invoice — $4,200",
            "body": (
                "Hi Billing Team,\n\n"
                "We were double-charged on invoice INV-2024-1892. Two identical charges "
                "of $4,200 appeared on our corporate card on Jan 15. Our CFO is asking "
                "for an immediate refund of the duplicate charge.\n\n"
                "Invoice reference: INV-2024-1892\n"
                "Card ending: ****3847\n"
                "Amount: $4,200.00 x2\n\n"
                "Please resolve this today. We need a confirmation email for our records.\n\n"
                "Thanks,\nFatima Al-Rashid\nFinance Director, PetroMax"
            ),
            "priority": "urgent",
            "action": "escalate",
            "reply_keywords": ["refund", "invoice", "billing", "resolved", "duplicate"],
            "thread_length": 1,
            "has_attachment": True,
        },
        # HIGH
        {
            "subject": "Can't update our payment method — getting errors",
            "body": (
                "Hello,\n\n"
                "I'm trying to update our payment method in the Billing settings but "
                "keep getting 'Payment method update failed' error. Our current card "
                "expires this week and I need to add the new one ASAP to avoid service "
                "interruption.\n\n"
                "I've tried 3 different cards — same error each time.\n\n"
                "Can someone help? This is time-sensitive.\n\n"
                "Thanks,\nCarlos Mendez"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["payment", "method", "update", "billing", "fix", "workaround"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Need a detailed usage breakdown for our Q4 invoice",
            "body": (
                "Hi,\n\n"
                "Our finance team needs a detailed breakdown of our Q4 2024 usage for "
                "budget reconciliation. The invoice total of $12,400 seems higher than "
                "expected and we need to understand the line items.\n\n"
                "Can you provide:\n"
                "- Per-user breakdown\n"
                "- Data ingestion volume\n"
                "- Any overage charges\n\n"
                "We need this by end of week for our quarterly review.\n\n"
                "Thanks,\nOlga Kuznetsova"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["invoice", "breakdown", "usage", "billing", "charges"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # MEDIUM
        {
            "subject": "Question about annual vs monthly pricing",
            "body": (
                "Hi,\n\n"
                "We're currently on the monthly Growth plan at $299/month. Considering "
                "switching to annual billing. What's the discount for annual commitment? "
                "Also, can we switch mid-cycle or do we have to wait?\n\n"
                "Thanks,\nAmara Obi"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["annual", "pricing", "discount", "billing", "plan"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Requesting W-9 form for vendor setup",
            "body": (
                "Hello CloudMetrics,\n\n"
                "Our accounts payable department needs a completed W-9 form to set you "
                "up as a vendor in our system. Can you send one over?\n\n"
                "Also, do you accept payment via ACH/wire transfer?\n\n"
                "Thanks,\nMichael O'Brien\nBigBank Procurement"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["W-9", "vendor", "payment", "ACH", "wire"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # LOW
        {
            "subject": "Receipt request for January payment",
            "body": (
                "Hi, could you resend the receipt for our January payment? I think "
                "I accidentally deleted the email. Need it for expense tracking.\n\n"
                "Account: Lisbon Tech\nAmount: ~$149\n\nThanks,\nSofia Costa"
            ),
            "priority": "low",
            "action": "reply",
            "reply_keywords": ["receipt", "invoice", "resend", "payment"],
            "thread_length": 1,
            "has_attachment": False,
        },
    ]
    return templates


def _support_emails(rng: random.Random) -> list[dict]:
    """Generate general support emails."""
    templates = [
        # HIGH
        {
            "subject": "Need help migrating data from our old analytics tool",
            "body": (
                "Hi team,\n\n"
                "We're migrating from Mixpanel to CloudMetrics and need guidance on "
                "importing our historical data. We have 2 years of event data (~50GB) "
                "in JSON format.\n\n"
                "Questions:\n"
                "1. What's the recommended import method for this volume?\n"
                "2. Is there a Mixpanel-specific migration guide?\n"
                "3. Can we map our existing event schemas to your data model?\n\n"
                "We'd love to schedule a migration call with your team if possible.\n\n"
                "Thanks,\nRaj Patel\nCTO, Initech"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["migration", "import", "data", "guide", "call", "schedule"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Our team can't access shared dashboards",
            "body": (
                "Hello,\n\n"
                "We created several shared dashboards last week, but now team members "
                "are getting 'Permission Denied' when trying to view them. I'm the org "
                "admin and I've verified the sharing settings look correct.\n\n"
                "This is affecting 12 team members who rely on these dashboards daily.\n\n"
                "Can you look into this?\n\nThanks,\nEmily Zhao"
            ),
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["dashboard", "permission", "sharing", "access", "fix"],
            "thread_length": 2,
            "has_attachment": False,
        },
        # MEDIUM
        {
            "subject": "How to set up custom event tracking?",
            "body": (
                "Hi,\n\n"
                "We want to track custom events beyond page views — specifically user "
                "interactions like button clicks, form submissions, and video plays.\n\n"
                "I've read the docs but I'm a bit confused about the JavaScript SDK "
                "initialization. Do I need to call `cloudmetrics.init()` on every page "
                "or just once in the app shell?\n\n"
                "Any code examples would be super helpful.\n\n"
                "Thanks,\nTom Bradley"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["custom events", "tracking", "SDK", "documentation", "code"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Best practices for organizing dashboards?",
            "body": (
                "Hi CloudMetrics,\n\n"
                "Our team has created ~40 dashboards and it's getting hard to find things. "
                "Do you have any recommended folder structures or naming conventions?\n\n"
                "Also, is there a way to tag or categorize dashboards?\n\n"
                "Thanks for any advice!\nLisa Nakamura"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["dashboards", "organize", "folders", "best practices"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Question about data retention policy",
            "body": (
                "Hello,\n\n"
                "We're conducting a GDPR compliance review and need to understand "
                "CloudMetrics' data retention policy. Specifically:\n\n"
                "1. How long is event data stored by default?\n"
                "2. Can we configure custom retention periods?\n"
                "3. Is there a data deletion API for GDPR 'right to erasure' requests?\n\n"
                "Please advise.\n\nIngrid Larsson\nDPO, NordicSaaS"
            ),
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["GDPR", "data retention", "compliance", "policy", "deletion"],
            "thread_length": 1,
            "has_attachment": False,
        },
        # LOW
        {
            "subject": "Do you have a community forum or Discord?",
            "body": (
                "Hey! Love CloudMetrics so far. Was wondering if you have a community "
                "forum, Discord, or Slack group where users can share tips and dashboards? "
                "Would be great to connect with other users.\n\nThanks!\nDavid Kim"
            ),
            "priority": "low",
            "action": "reply",
            "reply_keywords": ["community", "forum", "Discord", "connect"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Where's the status page?",
            "body": (
                "Quick question — do you have a public status page where I can check "
                "service health and subscribe to incident updates?\n\nThanks!\nYuki"
            ),
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["status page", "health", "incidents"],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Thanks for the great onboarding experience",
            "body": (
                "Just wanted to say thanks to whoever designed the onboarding flow. "
                "We had our first dashboard up and running in under 30 minutes. Great "
                "documentation and the interactive tutorial was really helpful.\n\n"
                "Keep up the good work!\n\nJean-Pierre Dubois"
            ),
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["thanks", "onboarding", "feedback"],
            "thread_length": 1,
            "has_attachment": False,
        },
    ]
    return templates


def _spam_emails(rng: random.Random) -> list[dict]:
    """Generate spam / phishing emails."""
    templates = [
        {
            "subject": "You've won a $1,000 Amazon Gift Card!",
            "body": (
                "CONGRATULATIONS! You have been selected as today's lucky winner!\n\n"
                "Click here to claim your $1,000 Amazon Gift Card: "
                "http://fr33-amaz0n-gift.xyz/claim?id=8271\n\n"
                "This offer expires in 24 hours. Act now!\n\n"
                "To unsubscribe, click here."
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Your account has been compromised — verify now",
            "body": (
                "Dear CloudMetrics User,\n\n"
                "We detected unusual activity on your account. To prevent unauthorized "
                "access, please verify your identity immediately by clicking the link below:\n\n"
                "https://securealert-update.click/verify?token=abc123\n\n"
                "If you do not verify within 12 hours, your account will be suspended.\n\n"
                "CloudMetrics Security Team"
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Exclusive Investment Opportunity — 500% ROI Guaranteed",
            "body": (
                "Hello,\n\n"
                "I represent a private investment fund with guaranteed annual returns "
                "of 500%. Our proprietary AI-driven trading algorithm has never posted "
                "a losing quarter.\n\n"
                "Minimum investment: $10,000\n"
                "Expected return: $50,000+ in 12 months\n\n"
                "Serious investors only. Reply with 'INTERESTED' for our prospectus.\n\n"
                "Dr. Robert\nChief Investment Officer"
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": True,
        },
        {
            "subject": "Re: Your LinkedIn connection request",
            "body": (
                "Hi,\n\n"
                "You have 3 pending connection requests on LinkedIn. Click below to "
                "view and accept:\n\n"
                "https://linkedln-verify.info/connections\n\n"
                "Note: This message was sent to cloudmetrics-support@cloudmetrics.com. "
                "If you believe this was sent in error, please disregard.\n\n"
                "LinkedIn Team"
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Apple ID Locked — Immediate Action Required",
            "body": (
                "Dear Customer,\n\n"
                "Your Apple ID has been locked due to suspicious activity. To unlock "
                "your account, please visit:\n\n"
                "https://apple-id-secure.top/unlock\n\n"
                "Failure to act within 24 hours will result in permanent account closure.\n\n"
                "Apple Support\n"
                "Cupertino, CA"
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "🚀 Crypto signals — Join 10,000+ winners",
            "body": (
                "Join our VIP crypto signals group and make $5K/day from home!\n\n"
                "✅ 95% accuracy rate\n"
                "✅ Real-time buy/sell signals\n"
                "✅ Free 7-day trial\n\n"
                "Join now: https://10xgains-crypto.io/vip\n\n"
                "Testimonial: 'I made $80K in my first month!' —John D."
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
        },
        {
            "subject": "Updated Employee Benefits Package — Action Required",
            "body": (
                "Dear Team Member,\n\n"
                "As part of our annual benefits update, we need you to confirm your "
                "benefits selections for 2025. Please download and complete the attached "
                "form by January 31.\n\n"
                "Form link: https://company-benefits-update.net/form\n\n"
                "Thank you,\nHR Department"
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": True,
        },
        {
            "subject": "Free Exclusive Masterclass: 10X Your Revenue",
            "body": (
                "Hey there,\n\n"
                "I'm hosting a FREE masterclass on how to 10X your SaaS revenue in 90 "
                "days. Only 50 spots available!\n\n"
                "In this masterclass you'll learn:\n"
                "- The #1 growth hack that doubled our ARR\n"
                "- How to get 100 customers in 30 days\n"
                "- The secret pricing strategy nobody talks about\n\n"
                "Register: https://exclusive-masterclass.guru/register\n\n"
                "See you there!\nJake \"The Growth Guy\" Thompson"
            ),
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
        },
    ]
    return templates


def _adversarial_emails(rng: random.Random) -> list[dict]:
    """
    Generate adversarial / tricky emails designed to confuse triage.

    These mix misleading signals: urgent-sounding subjects that are low-priority,
    spam disguised as legitimate, genuinely urgent issues buried in casual tone, etc.
    """
    templates = [
        # Misleading: URGENT subject but actually low priority
        {
            "subject": "URGENT!!!! PLEASE READ IMMEDIATELY!!!!",
            "body": (
                "Hey!\n\n"
                "Urgent reminder: Our company picnic is next Saturday at Riverside Park! "
                "Don't forget to RSVP by Thursday. We need a headcount for catering.\n\n"
                "Also, if you're bringing a dish, please sign up on the shared doc.\n\n"
                "See you there!\nCasey Morgan"
            ),
            "sender_override": ("Casey Morgan", "cloudmetrics.com"),
            "priority": "low",
            "category": "support",
            "action": "archive",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
        # Misleading: Casual tone but actually urgent
        {
            "subject": "Quick question about our account",
            "body": (
                "Hey there,\n\n"
                "So I noticed something kinda weird — all our user data from the last "
                "3 months seems to have vanished from the analytics dashboard. Like, "
                "completely gone. All the charts show zero after October 1.\n\n"
                "I'm not super technical but our CEO is asking about the Q4 numbers "
                "for an all-hands tomorrow at 9 AM. Is there any way to get this back? "
                "I hope it's not permanently deleted or anything 😅\n\n"
                "No rush if it's complicated, but yeah... our CEO is kinda freaking out.\n\n"
                "Thanks!\nAmara"
            ),
            "priority": "urgent",
            "category": "bug_report",
            "action": "escalate",
            "reply_keywords": ["data", "investigating", "urgent", "recovery", "engineering"],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
        # Disguised spam: Looks like a partnership inquiry
        {
            "subject": "Partnership Opportunity — CloudMetrics x DataSync",
            "body": (
                "Dear CloudMetrics Business Development Team,\n\n"
                "I'm reaching out from DataSync Solutions. We've been following your "
                "growth and believe there's a strong synergy between our platforms.\n\n"
                "We'd love to explore a co-marketing partnership. To get started, please "
                "fill out our partner onboarding form:\n\n"
                "https://datasync-partners.biz/onboard?ref=cloudmetrics\n\n"
                "I've attached our partnership deck for your review.\n\n"
                "Looking forward to hearing from you!\n\n"
                "Best regards,\nJake Williams\nVP Partnerships, DataSync Solutions"
            ),
            "sender_override": ("Jake Williams", "datasync-partners.biz"),
            "priority": "low",
            "category": "spam",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": True,
            "adversarial": True,
        },
        # Genuine escalation buried in a thread
        {
            "subject": "Re: Re: Re: Minor dashboard cosmetic issue",
            "body": (
                "Hi again,\n\n"
                "Okay, so the cosmetic thing I mentioned last week is now way worse. "
                "The dashboard isn't just displaying wrong colors — it's now showing "
                "OTHER CUSTOMERS' DATA in our dashboards.\n\n"
                "I can see company names, revenue figures, and user counts that are "
                "definitely not ours. This is a MAJOR security/privacy breach.\n\n"
                "I've taken screenshots but I'm not going to attach them for obvious "
                "data protection reasons.\n\n"
                "This needs to be handled IMMEDIATELY.\n\n"
                "—Marcus Johnson\nCISO, Wayne Enterprises"
            ),
            "priority": "urgent",
            "category": "bug_report",
            "action": "escalate",
            "reply_keywords": ["security", "breach", "data", "investigating", "immediate"],
            "thread_length": 5,
            "has_attachment": False,
            "adversarial": True,
        },
        # Looks urgent but is actually just a sales pitch
        {
            "subject": "IMPORTANT: Your cybersecurity is at risk",
            "body": (
                "Dear CloudMetrics Team,\n\n"
                "Recent industry reports show that 73% of SaaS companies experienced "
                "a data breach in 2024. Is your organization protected?\n\n"
                "CyberShield Pro offers enterprise-grade security monitoring starting "
                "at just $499/month. Our platform detects threats in real-time and "
                "provides automated remediation.\n\n"
                "Schedule a free 30-minute security assessment today:\n"
                "https://cybershield-pro.com/assessment\n\n"
                "Don't wait until it's too late.\n\n"
                "Robert Chen\nAccount Executive, CyberShield Pro"
            ),
            "sender_override": ("Robert Chen", "cybershield-pro.com"),
            "priority": "low",
            "category": "spam",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
        # Internal request that seems low-priority but has real urgency
        {
            "subject": "FYI — some billing thing",
            "body": (
                "Hey team,\n\n"
                "Just a heads up — I noticed our Stripe webhook endpoint is returning "
                "errors for subscription renewal events. Looks like about 200 customers' "
                "renewals failed silently over the weekend.\n\n"
                "These are Enterprise customers paying $500-$2000/month. If we don't fix "
                "the webhook and manually retry, they'll get cancellation emails tomorrow "
                "from the automated system.\n\n"
                "I set up a temporary workaround but it'll only hold for ~12 hours.\n\n"
                "—Sam Nguyen"
            ),
            "sender_override": ("Sam Nguyen", "cloudmetrics.com"),
            "priority": "urgent",
            "category": "billing",
            "action": "escalate",
            "reply_keywords": ["billing", "webhook", "Stripe", "urgent", "fix", "renewals"],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
        # Polite but actually needs escalation
        {
            "subject": "No worries if you're busy — small compliance question",
            "body": (
                "Hi CloudMetrics team,\n\n"
                "Apologies for bothering you — I know you're probably swamped! No rush "
                "at all, but our SOC 2 auditor has a few tiny questions about your data "
                "processing agreement.\n\n"
                "Actually, I just realized the audit deadline is tomorrow at 5 PM EST. "
                "If we don't have compliant responses by then, we'll fail the audit and "
                "have to pause our CloudMetrics integration per our legal team's policy. "
                "That would affect our entire product analytics.\n\n"
                "Could someone from your compliance team possibly respond? Again, so "
                "sorry for the inconvenience!\n\n"
                "Warmest regards,\nIngrid Larsson\nDPO, NordicSaaS"
            ),
            "priority": "urgent",
            "category": "support",
            "action": "escalate",
            "reply_keywords": ["SOC 2", "compliance", "audit", "urgent", "deadline"],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
        # Feature request that sounds like a bug
        {
            "subject": "BUG: Can't filter by custom date range",
            "body": (
                "Hi,\n\n"
                "I'm filing this as a bug because I think it should work but doesn't. "
                "When I go to Analytics > Reports, there's no option to set a custom "
                "date range. I can only select preset ranges like 'Last 7 days' or "
                "'Last 30 days'.\n\n"
                "I need to filter for a specific quarter (Oct 1 - Dec 31 2024). Is "
                "this feature missing or am I looking in the wrong place?\n\n"
                "If it's not implemented yet, I'd like to request it as a feature.\n\n"
                "Not urgent but would be really handy.\n\n"
                "Thanks,\nTom Bradley"
            ),
            "priority": "medium",
            "category": "feature_request",
            "action": "reply",
            "reply_keywords": ["date range", "custom", "feature", "roadmap", "workaround"],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
        # Legitimate-looking but is actually spam
        {
            "subject": "Re: Your support ticket #CM-45921",
            "body": (
                "Hi,\n\n"
                "Thank you for contacting CloudMetrics Support. We've reviewed your "
                "ticket and found an issue with your account configuration.\n\n"
                "To apply the fix, please verify your account credentials at:\n\n"
                "https://cloudmetrics-support-verify.info/ticket/45921\n\n"
                "This verification is required to apply the security patch to your "
                "account within 48 hours.\n\n"
                "Best,\nCloudMetrics Support Team"
            ),
            "sender_override": ("CloudMetrics Support", "cloudmetrics-support-verify.info"),
            "priority": "low",
            "category": "spam",
            "action": "delete",
            "reply_keywords": [],
            "thread_length": 2,
            "has_attachment": False,
            "adversarial": True,
        },
        # Looks like spam but is actually a high-value customer
        {
            "subject": "CONGRATULATIONS — You've been selected!!!",
            "body": (
                "Dear CloudMetrics,\n\n"
                "CONGRATULATIONS! Your company has been selected as a finalist for the "
                "2025 SaaS Innovation Award by TechCrunch!\n\n"
                "As a finalist, you're invited to present at our annual ceremony in "
                "San Francisco on March 15. The award includes $50,000 in AWS credits "
                "and a feature article on TechCrunch.\n\n"
                "To confirm your participation, please reply to this email by February 1. "
                "Note: We sent this to your support address because we couldn't find "
                "a direct contact for your founding team.\n\n"
                "Regards,\nSarah Chen\nSenior Editor, TechCrunch\nsarah@techcrunch.com"
            ),
            "sender_override": ("Sarah Chen", "techcrunch.com"),
            "priority": "high",
            "category": "support",
            "action": "reply",
            "reply_keywords": ["award", "interested", "confirm", "forward", "team"],
            "thread_length": 1,
            "has_attachment": False,
            "adversarial": True,
        },
    ]
    return templates


# ---------------------------------------------------------------------------
# Extra filler emails to reach 200 total
# ---------------------------------------------------------------------------


def _extra_emails(rng: random.Random) -> list[dict]:
    """Generate additional emails to pad the dataset to 200."""
    subjects_and_bodies = [
        # -- Bug reports --
        {
            "subject": "Graph rendering is slow on large datasets",
            "body": (
                "Hi,\n\nWhen we load dashboards with 100K+ data points, the chart "
                "rendering takes 15-20 seconds. It used to be under 3 seconds. "
                "Noticed this after the v4.3 update.\n\nBrowser: Chrome 121\n\nDavid"
            ),
            "category": "bug_report",
            "priority": "medium",
            "action": "archive",
            "reply_keywords": ["performance", "rendering", "investigating"],
        },
        {
            "subject": "Email digest arriving at wrong time",
            "body": (
                "I set my daily digest to arrive at 8 AM EST but it keeps coming at "
                "3 AM. I've checked my timezone settings and they're correct.\n\n"
                "Not a big deal, just wanted to flag it.\n\nRaj"
            ),
            "category": "bug_report",
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["digest", "timezone", "fix"],
        },
        {
            "subject": "Alert rules not firing for custom metrics",
            "body": (
                "We set up alert rules for our custom metrics (cpu_usage_p99 > 90%) "
                "but they haven't fired despite the metric clearly exceeding the "
                "threshold multiple times today. Standard metrics alerts work fine.\n\n"
                "This is impacting our incident response. Can you check?\n\nWei"
            ),
            "category": "bug_report",
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["alerts", "custom metrics", "investigating", "fix"],
        },
        {
            "subject": "Broken link in the API documentation",
            "body": (
                "The link to the 'Authentication' section in your API docs "
                "(https://docs.cloudmetrics.com/api/auth) returns a 404.\n\n"
                "Minor thing — just wanted to let you know.\n\nSofia"
            ),
            "category": "bug_report",
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["documentation", "link", "fix"],
        },
        {
            "subject": "Data discrepancy between API and dashboard",
            "body": (
                "Hi,\n\nI'm getting different numbers when I query the API directly "
                "versus what the dashboard shows for the same time range and metric. "
                "The API returns 14,523 events while the dashboard shows 14,891.\n\n"
                "Difference is ~2.5%. Is there a caching or aggregation issue?\n\n"
                "This is important because we use API data for billing customers.\n\n"
                "Marcus"
            ),
            "category": "bug_report",
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["data", "discrepancy", "API", "dashboard", "investigating"],
        },
        # -- Feature requests --
        {
            "subject": "Request: Multi-tenant dashboard support",
            "body": (
                "We're a white-label partner and need to create separate dashboard "
                "views for each of our customers. Currently we can only have one "
                "org-wide view.\n\n"
                "Would be great to have tenant isolation within a single account.\n\n"
                "Priya"
            ),
            "category": "feature_request",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["multi-tenant", "dashboard", "roadmap", "feature"],
        },
        {
            "subject": "Feedback: Would love real-time collaboration",
            "body": (
                "Love the product! One thing that would make it amazing for our team "
                "is real-time collaboration on dashboards — like Google Docs but for "
                "analytics. Seeing other users' cursors and live edits.\n\n"
                "I know it's a big ask but it would be game-changing.\n\n"
                "Carlos"
            ),
            "category": "feature_request",
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["collaboration", "real-time", "feedback", "roadmap"],
        },
        {
            "subject": "API: Need GraphQL endpoint support",
            "body": (
                "Hi team,\n\nAny plans to add a GraphQL API alongside the REST API? "
                "Our frontend is built with Apollo Client and a GraphQL endpoint would "
                "simplify our integration significantly.\n\nThanks,\nYuki"
            ),
            "category": "feature_request",
            "priority": "medium",
            "action": "archive",
            "reply_keywords": ["GraphQL", "API", "roadmap"],
        },
        {
            "subject": "Can we schedule report deliveries?",
            "body": (
                "Hi,\n\nIt would be great to schedule automatic report deliveries — "
                "e.g., send a specific dashboard view as a PDF to a distribution list "
                "every Monday at 9 AM.\n\n"
                "Several team members have asked for this.\n\nOlga"
            ),
            "category": "feature_request",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["scheduled", "report", "delivery", "feature", "roadmap"],
        },
        {
            "subject": "Suggestion: Natural language query for analytics",
            "body": (
                "What if users could type questions like 'What was our conversion rate "
                "last week?' and get charts automatically? Like an AI assistant for "
                "analytics.\n\nJust throwing the idea out there.\n\nJean-Pierre"
            ),
            "category": "feature_request",
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["NLP", "AI", "query", "feature"],
        },
        # -- Billing --
        {
            "subject": "Need to add 20 more seats midcycle",
            "body": (
                "Hi billing team,\n\nWe just hired a batch of new analysts and need "
                "to add 20 seats to our account immediately. How does prorated billing "
                "work for midcycle additions?\n\n"
                "Current plan: Growth (50 seats)\nNeeded: 70 seats\n\nFatima"
            ),
            "category": "billing",
            "priority": "high",
            "action": "reply",
            "reply_keywords": ["seats", "billing", "prorated", "upgrade"],
        },
        {
            "subject": "Tax exemption documentation",
            "body": (
                "Hello,\n\nWe're a non-profit educational institution and should be "
                "tax-exempt. I've attached our tax exemption certificate. Can you apply "
                "this to our account going forward?\n\nAmara"
            ),
            "category": "billing",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["tax", "exemption", "certificate", "applied"],
        },
        {
            "subject": "Can we get a PO-based billing arrangement?",
            "body": (
                "Hi,\n\nOur company requires purchase order-based billing rather than "
                "credit card charges. Is this available on the Enterprise plan?\n\n"
                "We'd need NET 30 payment terms.\n\nMichael"
            ),
            "category": "billing",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["purchase order", "PO", "billing", "NET 30", "enterprise"],
        },
        {
            "subject": "Downgrading our plan next month",
            "body": (
                "Hi,\n\nWe'd like to downgrade from Growth to Starter plan starting "
                "next billing cycle. Can you confirm what features we'll lose and "
                "whether our data will be preserved?\n\nTom"
            ),
            "category": "billing",
            "priority": "low",
            "action": "reply",
            "reply_keywords": ["downgrade", "plan", "features", "data"],
        },
        # -- Support --
        {
            "subject": "Need help with API key rotation",
            "body": (
                "Hi,\n\nWe need to rotate our API keys as part of a security audit. "
                "The docs mention a key rotation endpoint but I can't find it. Also, "
                "will rotating keys invalidate existing sessions?\n\nEmily"
            ),
            "category": "support",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["API key", "rotation", "security", "documentation"],
        },
        {
            "subject": "Onboarding 3 new team members — best approach?",
            "body": (
                "We have 3 new data analysts joining Monday. What's the best way to "
                "onboard them? Any training resources or recommended first steps?\n\n"
                "We want them productive within their first week.\n\nLisa"
            ),
            "category": "support",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["onboarding", "training", "resources", "team"],
        },
        {
            "subject": "SDK version compatibility question",
            "body": (
                "Hi,\n\nWe're using the Python SDK v2.3.1 with Python 3.12. Are there "
                "any known compatibility issues? We're seeing sporadic ImportErrors "
                "in our CI pipeline.\n\nDavid"
            ),
            "category": "support",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["SDK", "compatibility", "Python", "version"],
        },
        {
            "subject": "How to implement funnel analysis?",
            "body": (
                "Hello,\n\nI'm trying to set up a conversion funnel tracking "
                "sign-up → onboarding → first-dashboard → paid-conversion but I'm "
                "not sure how to define the funnel steps in CloudMetrics.\n\n"
                "Any guides or examples?\n\nSofia"
            ),
            "category": "support",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["funnel", "analysis", "documentation", "guide", "steps"],
        },
        {
            "subject": "Integrating CloudMetrics with Segment",
            "body": (
                "We use Segment as our CDP and want to send events to CloudMetrics "
                "via the Segment destination. Is there an official integration or do "
                "we need to use the HTTP API destination?\n\nIngrid"
            ),
            "category": "support",
            "priority": "medium",
            "action": "reply",
            "reply_keywords": ["Segment", "integration", "CDP", "destination"],
        },
        {
            "subject": "Curious about your tech stack",
            "body": (
                "Hey! I'm a big fan of CloudMetrics and curious what your tech stack "
                "looks like under the hood. ClickHouse? TimescaleDB? Just curious, "
                "no expectations for a detailed answer.\n\nJean-Pierre"
            ),
            "category": "support",
            "priority": "low",
            "action": "archive",
            "reply_keywords": ["tech stack", "architecture"],
        },
        # -- More spam for variety --
        {
            "subject": "Get 10,000 Instagram followers overnight!",
            "body": (
                "Want to grow your social media presence FAST?\n\n"
                "Our service delivers 10,000 real, active Instagram followers in just "
                "24 hours. 100% safe, no bots, guaranteed.\n\n"
                "Use code BOOST50 for 50% off: https://insta-booster.xyz\n\n"
                "Results or your money back!"
            ),
            "category": "spam",
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
        },
        {
            "subject": "Meeting tomorrow — important docs attached",
            "body": (
                "Hi,\n\n"
                "Please review the attached documents before our meeting tomorrow. "
                "It is very important that you download and open them.\n\n"
                "Download link: https://docs-download-secure.xyz/meeting-docs.zip\n\n"
                "Best regards,\nAdministrator"
            ),
            "category": "spam",
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
        },
        {
            "subject": "SEO Services — Get Page 1 on Google Guaranteed",
            "body": (
                "Hello CloudMetrics Marketing Team,\n\n"
                "Is your website not ranking on Google? We guarantee first page rankings "
                "within 30 days or your money back.\n\n"
                "Our proven SEO strategy has helped 500+ businesses achieve top rankings. "
                "Starting at just $99/month.\n\n"
                "Reply 'YES' for a free audit.\n\n"
                "Best,\nDigital Marketing Pro Team"
            ),
            "category": "spam",
            "priority": "low",
            "action": "delete",
            "reply_keywords": [],
        },
    ]

    senders_by_category = {
        "bug_report": _LEGIT_SENDERS,
        "feature_request": _LEGIT_SENDERS,
        "billing": _LEGIT_SENDERS,
        "support": _LEGIT_SENDERS,
        "spam": _SPAM_SENDERS,
    }

    emails: list[dict] = []
    for t in subjects_and_bodies:
        cat = t["category"]
        pool = senders_by_category[cat]
        sender_name, sender_domain = rng.choice(pool)
        emails.append(
            {
                "subject": t["subject"],
                "body": t["body"],
                "sender_name": sender_name,
                "sender_domain": sender_domain,
                "priority": t["priority"],
                "category": cat,
                "action": t["action"],
                "reply_keywords": t.get("reply_keywords", []),
                "thread_length": rng.randint(1, 3),
                "has_attachment": rng.random() < 0.15,
            }
        )
    return emails


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

_BASE_TIMESTAMP = "2025-01-20T08:00:00Z"


def _make_timestamp(rng: random.Random, index: int) -> str:
    """Generate a plausible ISO 8601 timestamp for email `index`."""
    # Emails arrive over a ~5-day window, roughly in order
    hour = 8 + (index * 37 + rng.randint(0, 12)) % 14  # 08:00–21:59
    minute = rng.randint(0, 59)
    day = 20 + index // 40  # spread across Jan 20–24
    day = min(day, 24)
    return f"2025-01-{day:02d}T{hour:02d}:{minute:02d}:00Z"


def generate_email_dataset(seed: int = 42) -> list[GroundTruthEmail]:
    """
    Generate the full synthetic dataset of 200 emails.

    Returns a deterministic list of GroundTruthEmail objects.
    Ground-truth labels are baked in and never shown to the agent.
    """
    rng = random.Random(seed)

    all_emails: list[GroundTruthEmail] = []
    idx = 0

    # ---- Category-specific templates ----
    category_generators = [
        ("bug_report", _bug_reports),
        ("feature_request", _feature_requests),
        ("billing", _billing_emails),
        ("support", _support_emails),
        ("spam", _spam_emails),
    ]

    for category, generator in category_generators:
        templates = generator(rng)
        for t in templates:
            sender_name, sender_domain = rng.choice(
                _SPAM_SENDERS if category == "spam" else _LEGIT_SENDERS
            )
            all_emails.append(
                GroundTruthEmail(
                    email_id=f"email_{idx:04d}",
                    subject=t["subject"],
                    sender=sender_name,
                    sender_domain=sender_domain,
                    body=t["body"],
                    timestamp=_make_timestamp(rng, idx),
                    thread_length=t.get("thread_length", 1),
                    has_attachment=t.get("has_attachment", False),
                    true_priority=t["priority"],
                    true_category=category,
                    expected_action=t["action"],
                    reply_keywords=t.get("reply_keywords", []),
                    is_adversarial=False,
                )
            )
            idx += 1

    # ---- Adversarial emails ----
    for t in _adversarial_emails(rng):
        sender_info = t.get("sender_override")
        if sender_info:
            sender_name, sender_domain = sender_info
        else:
            sender_name, sender_domain = rng.choice(_LEGIT_SENDERS)

        all_emails.append(
            GroundTruthEmail(
                email_id=f"email_{idx:04d}",
                subject=t["subject"],
                sender=sender_name,
                sender_domain=sender_domain,
                body=t["body"],
                timestamp=_make_timestamp(rng, idx),
                thread_length=t.get("thread_length", 1),
                has_attachment=t.get("has_attachment", False),
                true_priority=t["priority"],
                true_category=t["category"],
                expected_action=t["action"],
                reply_keywords=t.get("reply_keywords", []),
                is_adversarial=True,
            )
        )
        idx += 1

    # ---- Extra filler emails ----
    extras = _extra_emails(rng)
    for e in extras:
        all_emails.append(
            GroundTruthEmail(
                email_id=f"email_{idx:04d}",
                subject=e["subject"],
                body=e["body"],
                sender=e["sender_name"],
                sender_domain=e["sender_domain"],
                timestamp=_make_timestamp(rng, idx),
                thread_length=e.get("thread_length", 1),
                has_attachment=e.get("has_attachment", False),
                true_priority=e["priority"],
                true_category=e["category"],
                expected_action=e["action"],
                reply_keywords=e.get("reply_keywords", []),
                is_adversarial=False,
            )
        )
        idx += 1

    # ---- Duplicate / paraphrase some emails to reach exactly 200 ----
    while len(all_emails) < 200:
        # Clone a random non-spam email with slight variation
        source = rng.choice([e for e in all_emails if e.true_category != "spam"])
        variation_prefixes = [
            "Fwd: ",
            "Re: ",
            "Follow-up: ",
            "Update: ",
            "Reminder: ",
        ]
        new_subject = rng.choice(variation_prefixes) + source.subject
        new_body = source.body + "\n\n(Sent from my mobile device)"
        sender_name, sender_domain = rng.choice(_LEGIT_SENDERS)
        all_emails.append(
            GroundTruthEmail(
                email_id=f"email_{idx:04d}",
                subject=new_subject,
                sender=sender_name,
                sender_domain=sender_domain,
                body=new_body,
                timestamp=_make_timestamp(rng, idx),
                thread_length=source.thread_length + rng.randint(0, 2),
                has_attachment=source.has_attachment,
                true_priority=source.true_priority,
                true_category=source.true_category,
                expected_action=source.expected_action,
                reply_keywords=source.reply_keywords,
                is_adversarial=source.is_adversarial,
            )
        )
        idx += 1

    # Shuffle deterministically
    rng.shuffle(all_emails)

    # Re-index after shuffle
    for i, email in enumerate(all_emails):
        email.email_id = f"email_{i:04d}"

    return all_emails[:200]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_easy_emails(seed: int = 42) -> list[GroundTruthEmail]:
    """Return 20 non-adversarial, clear-signal emails for TASK 1."""
    rng = random.Random(seed + 1)
    pool = [e for e in generate_email_dataset(seed) if not e.is_adversarial]
    selected = rng.sample(pool, 20)
    for i, email in enumerate(selected):
        email.email_id = f"easy_{i:04d}"
    return selected


def get_medium_emails(seed: int = 42) -> list[GroundTruthEmail]:
    """
    Return 20 emails for TASK 2 (8 of which require replies).

    Ensures at least 8 emails have expected_action == 'reply'.
    """
    rng = random.Random(seed + 2)
    pool = [e for e in generate_email_dataset(seed) if not e.is_adversarial]
    reply_pool = [e for e in pool if e.expected_action == "reply"]
    non_reply_pool = [e for e in pool if e.expected_action != "reply"]

    # Pick exactly 8 reply-required emails
    replies = rng.sample(reply_pool, min(8, len(reply_pool)))
    # Fill with 12 non-reply emails
    others = rng.sample(non_reply_pool, 20 - len(replies))
    selected = replies + others
    rng.shuffle(selected)
    for i, email in enumerate(selected):
        email.email_id = f"medium_{i:04d}"
    return selected


def get_hard_emails(seed: int = 42) -> list[GroundTruthEmail]:
    """
    Return 30 emails for TASK 3 including adversarial cases.

    Includes adversarial emails plus a mix of regular ones.
    Ensures exactly 5 emails warrant escalation.
    """
    rng = random.Random(seed + 3)
    pool = generate_email_dataset(seed)
    adversarial = [e for e in pool if e.is_adversarial]
    regular = [e for e in pool if not e.is_adversarial]

    # Split adversarial into escalation vs non-escalation
    adv_escalation = [e for e in adversarial if e.expected_action == "escalate"]
    adv_non_escalation = [e for e in adversarial if e.expected_action != "escalate"]

    # Pick exactly 5 escalation-worthy emails from the adversarial set
    # (these are the tricky ones — casual tone but urgent, buried urgency, etc.)
    escalation_selected = rng.sample(adv_escalation, min(5, len(adv_escalation)))

    # If we need more escalation emails, grab from regular pool
    if len(escalation_selected) < 5:
        reg_escalation = [e for e in regular if e.expected_action == "escalate"]
        needed_esc = 5 - len(escalation_selected)
        escalation_selected.extend(
            rng.sample(reg_escalation, min(needed_esc, len(reg_escalation)))
        )

    # Start with all escalation picks + all non-escalation adversarial emails
    selected = list(escalation_selected) + list(adv_non_escalation)

    # Fill remaining spots with non-escalation regular emails
    used_ids = {e.email_id for e in selected}
    non_esc_regular = [
        e for e in regular
        if e.email_id not in used_ids and e.expected_action != "escalate"
    ]
    needed = 30 - len(selected)
    if needed > 0:
        selected.extend(rng.sample(non_esc_regular, min(needed, len(non_esc_regular))))

    rng.shuffle(selected)
    # Trim to exactly 30
    selected = selected[:30]
    for i, email in enumerate(selected):
        email.email_id = f"hard_{i:04d}"
    return selected
