Sports Autopsy 🏥📊
Overview
Sports Autopsy is an interactive digital forensic analysis system for athletic injuries that combines multidisciplinary expertise with local AI analysis using Ollama. This tool provides comprehensive injury assessment, identifies contributing factors, and generates personalized prevention plans.

Features
🎯 Core Capabilities
Interactive Data Collection: Guided input for athlete profiles, injury details, and training history

Multidisciplinary Analysis: Four expert perspectives on every injury

Counterfactual Scenarios: "What-if" analysis for injury prevention

Personalized Prevention Plans: Actionable, timeline-based rehabilitation strategies

Local AI Integration: Privacy-focused analysis using Ollama LLMs

Comprehensive Reporting: Detailed text reports with all analyses and recommendations

🔬 Analysis Perspectives
Sports Biomechanist: Movement pattern and technical breakdown analysis

Sports Physiotherapist: Tissue loading and recovery deficit assessment

Strength & Conditioning Coach: Training programming and physical preparation evaluation

Sports Psychologist: Psychological and behavioral factor analysis

Installation
Prerequisites
Python 3.8 or higher

Pip package manager

Step 1: Clone/Download
bash
git clone <repository-url>
cd sports-autopsy
Step 2: Install Python Dependencies
bash
pip install pandas numpy scikit-learn requests
Step 3: Install Ollama (Optional but Recommended)
bash
# Visit https://ollama.com/ for installation instructions
# Or use the automatic check in the script
Step 4: Download an Ollama Model
bash
ollama pull mistral
# or
ollama pull llama3.2
Usage
Basic Usage
bash
python injury_autopsy.py
Interactive Mode
The system will guide you through:

Athlete Information: Name, age, sport, position, experience

Injury Details: Type, severity, mechanism, date

Training Context: Load, recovery, sleep, nutrition

Medical History: Previous injuries, surgeries

Performance Metrics: Trends, fatigue, pain levels

Without Ollama
The system works in "template mode" without Ollama, providing structured but generic responses. For AI-powered analysis, install Ollama first.

Output
Console Display: All analyses shown in terminal

Text Report: Comprehensive report saved as injury_report_[name]_[timestamp].txt

System Architecture
Components
OllamaClient: Manages local LLM interactions

InteractiveInjuryAutopsy: Main analysis workflow controller

AthleteMetrics: Data structure for athlete metrics

Analysis Modules: Biomechanical, physiotherapy, S&C, psychological

Data Flow
text
User Input → Data Collection → Multidisciplinary Analysis → 
Counterfactual Scenarios → Prevention Plan → Final Report
AI Models Supported
Recommended Models
Mistral (Default): Good balance of speed and accuracy

Llama 3.2: Latest Meta model, excellent reasoning

Orca-mini: Smaller, faster alternative

Any Ollama-compatible model

Model Selection
The system automatically detects available models and uses the first one found. You can modify the default in the OllamaClient initialization.

Example Use Case
Scenario: Soccer Player Hamstring Strain
text
Athlete: Alex Johnson (24, Midfielder)
Injury: Grade 2 Hamstring Strain
Mechanism: Sprinting during match

System Provides:
1. Biomechanical: Reduced hip extension analysis
2. Physiotherapy: Load management failures
3. S&C: Strength imbalance recommendations
4. Psychology: Stress-injury relationship
5. Prevention: 12-week return-to-play plan
Customization
Modifying Analysis Prompts
Edit the generate_analysis_prompts() method to customize expert perspectives or add new analysis types.

Adding Metrics
Extend the AthleteMetrics dataclass to include additional tracking metrics relevant to your sport.

Model Parameters
Adjust temperature, max tokens, and system prompts in the OllamaClient.generate() method.

Benefits
For Coaches & Trainers
Evidence-based injury prevention strategies

Structured rehabilitation protocols

Multidisciplinary insights in one tool

For Athletes
Personalized recovery plans

Understanding of injury causation

Prevention strategies for future

For Sports Organizations
Consistent injury documentation

Data-driven prevention programs

Reduced injury recurrence rates

Limitations & Considerations
Current Limitations
Requires manual data input (no sensor integration)

Generic without sport-specific customization

Dependent on user-provided information accuracy

Privacy & Security
Local Processing: All analysis stays on your machine

No Cloud Dependencies: Works completely offline with Ollama

Data Control: You own all athlete data

Future Enhancements
Planned Features
CSV import for athlete metrics

Integration with wearable data

Sport-specific templates

Graphical visualization of risk factors

Web interface

Mobile app companion

Research Integration
Connection to sports science databases

Evidence-based recommendation weighting

Longitudinal tracking of prevention efficacy
