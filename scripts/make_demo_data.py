"""Generate realistic sample enterprise documents into data/docs/.

Covers every format the ingestion pipeline supports (.pdf, .txt, .md) so the
RAG system can be exercised end-to-end without needing real company data.

Usage:
    python scripts/make_demo_data.py
"""

from __future__ import annotations

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

import sys
from pathlib import Path

# Make `app` importable when running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

DOCS_DIR = settings.DOCS_DIR  # default tenant's docs folder

# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------

BENEFITS_PDF = (
    "Nexora Systems - Employee Benefits Program (2026)\n",
    "Overview\n"
    "Nexora Systems offers a comprehensive benefits package designed to support "
    "the health, financial wellbeing, and work-life balance of every employee. "
    "All regular full-time employees become eligible for benefits on the first "
    "day of the month following their start date. Part-time employees working "
    "at least twenty hours per week receive a reduced set of benefits including "
    "medical, dental, and vision coverage.\n"
    "Health Insurance\n"
    "Nexora provides three medical plan options through United Care: the PPO "
    "plan, the HMO plan, and the High Deductible Health Plan (HDHP). The PPO plan "
    "allows employees to visit any doctor without a referral and reimburses "
    "seventy percent of covered costs after the deductible. The HMO plan "
    "requires members to choose a primary care physician and referrals for "
    "specialists, but has lower monthly premiums and no deductible. The HDHP "
    "features the lowest premiums and works with a Health Savings Account, to "
    "which the company contributes one thousand five hundred dollars per year.\n"
    "Dental and Vision\n"
    "Dental coverage through BrightSmile includes two preventive cleanings and "
    "exams per year at no cost, with fifty percent coverage on basic procedures "
    "such as fillings and root canals. Orthodontics for dependents is covered up "
    "to a lifetime maximum of two thousand dollars. Vision coverage through "
    "EyeCare Partners provides one comprehensive eye exam and two hundred dollars "
    "toward frames and lenses every twelve months.\n"
    "Retirement Savings\n"
    "The company sponsors a 401(k) retirement plan through Fidelity. Employees "
    "may contribute up to the IRS annual limit, and Nexora matches one hundred "
    "percent of contributions up to four percent of salary, plus fifty percent "
    "of the next two percent. Contributions are immediately vested. Employees "
    "are automatically enrolled at a three percent contribution rate unless they "
    "opt out within sixty days.\n"
    "Paid Time Off and Leave\n"
    "Full-time employees receive twenty days of paid time off per year, which "
    "accrues on a pay-period basis and rolls over up to forty days. The company "
    "also provides ten paid holidays, twelve weeks of paid parental leave, and "
    "short-term disability coverage of one hundred percent of base salary for up "
    "to twelve weeks.\n"
    "Wellness Programs\n"
    "Nexora reimburses up to six hundred dollars per year for gym memberships, "
    "fitness classes, and wellness coaching. Annual preventive physicals are "
    "encouraged, and employees who complete the health assessment earn a two "
    "hundred dollar premium reduction on the following year's medical coverage.",
)

SECURITY_PDF = (
    "Nexora Systems - Information Security Policy (2026)\n",
    "Purpose and Scope\n"
    "This policy defines the rules for protecting Nexora Systems information "
    "assets, including company devices, data, and networks. It applies to all "
    "employees, contractors, and temporary workers who access company resources. "
    "Compliance with this policy is a condition of employment, and violations "
    "may result in disciplinary action up to and including termination.\n"
    "Password Requirements\n"
    "All accounts must use strong passwords of at least twelve characters "
    "containing a mix of uppercase letters, lowercase letters, digits, and "
    "symbols. Passwords must be unique across systems and must never be shared. "
    "Multi-factor authentication is mandatory for all accounts that access "
    "corporate email, source code repositories, or the VPN. Password managers "
    "approved by IT, such as KeySafe Pro, are provided to all employees.\n"
    "Device Security\n"
    "Company-issued laptops must remain encrypted with full-disk encryption at "
    "all times. Lock screens must activate after no more than five minutes of "
    "inactivity. Devices must receive operating system and antivirus updates "
    "within forty-eight hours of release. Personal devices are permitted for "
    "remote work only after enrolling in the mobile device management program.\n"
    "Data Classification and Handling\n"
    "Data is classified as Public, Internal, Confidential, or Restricted. "
    "Confidential data such as customer records and financial results must be "
    "stored only in approved systems, transmitted only over encrypted channels, "
    "and accessed only by personnel with a legitimate business need. Restricted "
    "data such as source code and merger information requires additional "
    "approval. Downloading confidential data to personal devices is forbidden.\n"
    "Network and Email Security\n"
    "Use of unapproved cloud storage services is prohibited. All remote access "
    "must go through the corporate VPN. Employees must report phishing attempts "
    "immediately to the security operations center by clicking the report button "
    "in Outlook. Software may only be installed from the approved corporate "
    "software catalog.\n"
    "Incident Response\n"
    "Any suspected security incident, including lost or stolen devices, "
    "suspicious emails, or unauthorized access, must be reported to the security "
    "team within one hour. The incident response team will contain the event, "
    "preserve evidence, and coordinate the investigation with the legal "
    "department. Employees are never penalized for promptly reporting incidents.",
)

ONBOARDING_TXT = """Nexora Systems - New Employee Onboarding Guide

Welcome to Nexora! This guide walks you through your first two weeks so you can
become productive quickly and understand how we work.

Day One: Orientation
Report to the Human Resources office in building A at nine in the morning. Bring
your photo ID for badge issuance. You will meet your manager, receive your
laptop and credentials, and complete your I-9 paperwork. Your manager will walk
you through the team's current projects and introduce you to your teammates.

Day Two: Access and Accounts
Once your account is active, IT will provision your email, calendar, and access
to the corporate systems listed in your welcome letter. Set up multi-factor
authentication on day one using the KeySafe Pro app. Install the company
password manager and store your credentials there. Never write passwords on
paper or in plain-text files.

Day Three to Five: Learning the Codebase
Our repositories live in the internal Git server under the "nexora" namespace.
Clone the main service repository and follow the README to run it locally. The
onboarding branch contains starter tasks labeled "good first issue". Schedule
a pairing session with your onboarding buddy, who is assigned on day one and is
your first point of contact for questions.

Week Two: Goals and Expectations
By the end of week two you should have your first small change merged to the
main branch. Your manager will meet with you to set your first-quarter goals in
the performance management system. Review the engineering standards wiki and
the code review checklist before submitting your first pull request.

Resources
The intranet has a knowledge base with setup guides, the engineering handbook,
and recorded onboarding sessions. If you are stuck, ask in the #onboarding
channel before searching on your own for more than thirty minutes. Human
Resources runs a weekly new-hire lunch every Friday at noon in the cafeteria.
"""

PRODUCT_MANUAL_MD = """# Nexora Atlas Cloud Gateway - Operator Manual

## Introduction

The Atlas Cloud Gateway is Nexora's enterprise API management platform. It
routes, secures, and observes API traffic between internal services and external
partners. This manual covers installation, configuration, and troubleshooting.

## System Requirements

Atlas runs on Linux servers or Kubernetes. Minimum requirements are four CPU
cores, eight gigabytes of RAM, and fifty gigabytes of disk. Production
deployments require a minimum of three gateway nodes behind a load balancer.
The gateway listens on port 8443 for HTTPS traffic and port 9090 for the admin
API.

## Installation

Deploy the gateway by applying the Helm chart included in the release bundle:

    helm install atlas ./charts/atlas-gateway --namespace gateway

The chart creates a StatefulSet, two services, and a ConfigMap. Before
installing, edit the values.yaml file to set your license key, DNS name, and
TLS certificate secret. Verify the deployment with the command `kubectl get pods`
and confirm all pods reach the Running state.

## Configuration

The main configuration file is `gateway.yaml`. The three most important
sections are routes, upstreams, and policies. Routes map incoming requests to
backend services. Upstreams define the target server pools with health-check
intervals and load balancing weights. Policies attach rate limiting, IP
allowlists, and request transformation rules.

Rate limiting is configured per route with a window size in seconds and a max
request count. Example: a route with a window of sixty seconds and a maximum of
one hundred requests allows one hundred requests per minute per client key.

## Monitoring

Metrics are exposed in Prometheus format at `/metrics`. Key indicators are
request latency, error rate, and active connections. Alerts should be
configured when the p95 latency exceeds one second or when the error rate
exceeds one percent over five minutes. Logs are written to stdout in JSON
format and should be shipped to the central log aggregator.

## Troubleshooting

If clients report timeouts, first check the upstream health status in the admin
API. If an upstream is marked unhealthy, restart the backend service and confirm
the health check endpoint returns a two hundred status. Certificate errors
indicate a missing or expired TLS secret; renew certificates at least thirty
days before expiry. For rate limit issues, verify the client key header is being
forwarded correctly, since rate limits are keyed on that header value.
"""

QUARTERLY_REPORT_TXT = """Nexora Systems - Q3 2026 Operations Report

Executive Summary
Revenue for the third quarter reached 42.1 million dollars, up 14 percent from
the previous quarter and 21 percent year over year. Operating expenses grew 8
percent to 33.4 million dollars, resulting in a net operating margin of 20.7
percent. The company added 112 new enterprise customers and ended the quarter
with a customer retention rate of 96 percent.

Revenue Breakdown
The Atlas platform generated 28.6 million dollars, or 68 percent of total
revenue, driven by strong demand for the new observability module. Professional
services contributed 8.3 million dollars, and maintenance renewals added 5.2
million dollars. International sales accounted for 34 percent of total revenue,
up from 28 percent in the prior year, with the strongest growth in the EMEA
region.

Customer Operations
The support organization handled 4,870 tickets during the quarter with a median
first-response time of 4 hours and a median resolution time of 22 hours. The
customer satisfaction score held steady at 4.7 out of 5. The top support issue
category was rate limiting configuration, representing 31 percent of tickets,
followed by certificate management at 22 percent. A knowledge base article
addressing the rate limiting topic was published in September and reduced
related ticket volume by 18 percent in the final month of the quarter.

Engineering and Delivery
Engineering shipped 14 releases across the platform with a mean lead time of
11 days per release and an on-time delivery rate of 93 percent. The incident
count remained low at 6 high-severity incidents, down from 9 in the previous
quarter. Mean time to restore for high-severity incidents was 74 minutes. Two
major initiatives launched: the multi-region deployment feature and the partner
developer portal.

Headcount and Operations
Headcount grew to 418 employees, up 12 percent from the start of the year.
Attrition remained below industry average at 9 percent annually. The company
opened a new office in Toronto in August to support the growing Canadian
customer base and expanded the security operations team from 5 to 9 members.

Outlook
For Q4, management expects revenue between 45 and 47 million dollars. Priorities
include scaling the partner developer portal, expanding the EMEA support desk,
and continuing the security hardening initiative. The board approved a 6 million
dollar investment in the observability roadmap for fiscal year 2027.
"""

# ---------------------------------------------------------------------------
# PDF generation helper
# ---------------------------------------------------------------------------


def write_pdf(path: Path, title: str, paragraphs: tuple[str, ...]) -> None:
    """Render a simple, clean multi-page PDF from plain-text paragraphs."""
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )
    styles = getSampleStyleSheet()
    story: list = [Paragraph(title, styles["Title"]), Spacer(1, 0.2 * inch)]
    for para in paragraphs:
        story.append(Paragraph(para.replace("\n", " "), styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))
    doc.build(story)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Demo documents will be written to {DOCS_DIR}")

    write_pdf(DOCS_DIR / "benefits_overview.pdf", "Employee Benefits Program (2026)", BENEFITS_PDF)
    write_pdf(DOCS_DIR / "security_policy.pdf", "Information Security Policy (2026)", SECURITY_PDF)

    (DOCS_DIR / "onboarding_guide.txt").write_text(ONBOARDING_TXT, encoding="utf-8")
    (DOCS_DIR / "atlas_gateway_manual.md").write_text(PRODUCT_MANUAL_MD, encoding="utf-8")
    (DOCS_DIR / "q3_2026_report.txt").write_text(QUARTERLY_REPORT_TXT, encoding="utf-8")

    print(f"Demo documents written to {DOCS_DIR}")


if __name__ == "__main__":
    main()
