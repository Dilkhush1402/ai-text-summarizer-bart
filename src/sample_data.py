"""
Curated sample articles and ground-truth reference summaries for testing and evaluation.
"""

from typing import Dict, List

SAMPLE_TEXTS: List[Dict[str, str]] = [
    {
        "id": "sample_ai",
        "title": "Artificial Intelligence in Modern Medicine",
        "category": "Technology & Healthcare",
        "text": (
            "Artificial intelligence (AI) is transforming modern healthcare through advanced diagnostic systems, "
            "automated medical imaging analysis, and personalized treatment recommendations. Deep learning models "
            "trained on millions of radiology scans have demonstrated accuracy matching or exceeding board-certified "
            "radiologists in identifying pulmonary nodules, diabetic retinopathy, and early-stage breast cancer. "
            "Beyond visual diagnosis, natural language processing models extract actionable clinical insights from "
            "unstructured electronic health records, accelerating rare disease identification and clinical trial matching. "
            "Moreover, generative algorithms assist pharmaceutical researchers by simulating molecular interactions and "
            "predicting protein folding structures, reducing the timeline of novel drug candidate discovery from years to "
            "months. Despite these remarkable technological leaps, widespread clinical deployment faces several critical "
            "challenges, including model interpretability, demographic bias in training corpora, patient privacy regulations, "
            "and the fundamental necessity for continuous physician-in-the-loop oversight to ensure patient safety and ethical integrity."
        ),
        "reference_summary": (
            "Artificial intelligence is revolutionizing healthcare diagnostics, imaging, electronic health records analysis, "
            "and drug discovery. However, clinical adoption requires resolving challenges regarding model interpretability, "
            "training bias, patient privacy, and clinician oversight."
        ),
    },
    {
        "id": "sample_quantum",
        "title": "Quantum Computing and Modern Cybersecurity",
        "category": "Computer Science & Cryptography",
        "text": (
            "Quantum computing represents a paradigm shift in computational power by exploiting principles of quantum "
            "mechanics, specifically superposition and quantum entanglement. Unlike classical computing systems that encode "
            "information into discrete binary bits representing zero or one, quantum processors operate on quantum bits "
            "or qubits, which can exist in linear superpositions of states. This capability provides exponential speedups "
            "for specific mathematical problems, including prime factorization through Shor's algorithm. Consequently, "
            "the emergence of fault-tolerant quantum hardware poses an existential threat to contemporary asymmetric "
            "cryptography standards, such as RSA and Elliptic Curve Cryptography, which secure modern banking, internet "
            "communications, and government infrastructure. In response to this looming security vulnerability, the National "
            "Institute of Standards and Technology (NIST) and international research groups have accelerated the standardization "
            "and deployment of post-quantum cryptography (PQC) based on lattice-based mathematics. Organizations worldwide "
            "are currently executing cryptographic transitions to safeguard sensitive historical and future data transmissions."
        ),
        "reference_summary": (
            "Quantum computing uses superposition and entanglement to solve specific mathematical problems exponentially faster "
            "than classical computers. This threatens traditional asymmetric encryption like RSA, prompting global efforts "
            "to standardize and deploy post-quantum lattice cryptography."
        ),
    },
    {
        "id": "sample_energy",
        "title": "The Global Transition to Renewable Smart Grids",
        "category": "Energy & Sustainability",
        "text": (
            "The global transition toward carbon-neutral power systems has catalyzed massive investments in solar photovoltaic, "
            "offshore wind, and utility-scale energy storage technologies. While renewable sources significantly reduce greenhouse "
            "gas emissions, their weather-dependent intermittency presents unprecedented operational challenges for legacy electricity "
            "transmission networks. Traditional grids were engineered around centralized, predictable fossil-fuel generation facilities "
            "that could readily throttle output according to consumer demand. In contrast, modern renewable systems require bidirectional "
            "power flows, distributed energy resource management, and high-frequency grid stabilization mechanisms. To address these "
            "dynamics, electrical utilities are deploying smart grid infrastructure integrated with artificial intelligence, smart meters, "
            "and industrial lithium-iron-phosphate battery reserves. These automated systems predict localized generation dips, optimize "
            "dynamic pricing incentives, and automatically route surplus renewable energy to avoid catastrophic blackouts."
        ),
        "reference_summary": (
            "The shift to renewable solar and wind power requires transitioning from centralized legacy grids to AI-enabled smart grids. "
            "Smart grids incorporate utility-scale battery storage and automated power routing to stabilize electricity networks against "
            "weather intermittency."
        ),
    },
]

