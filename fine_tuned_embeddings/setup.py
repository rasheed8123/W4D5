#!/usr/bin/env python3
"""
Setup script for Sales Conversion AI System
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models',
        'training',
        'pipeline',
        'ui',
        'evaluation_results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def download_sample_data():
    """Download or create sample data if not exists"""
    if os.path.exists('data/sample_transcripts.csv'):
        print("✅ Sample data already exists")
        return True
    
    print("📊 Creating sample data...")
    
    # Create sample transcripts CSV
    sample_data = [
        {
            'transcript_id': 1,
            'call_transcript': "Agent: Hi John, thanks for taking the time to speak with me today. I wanted to follow up on the demo we had last week. How did that go for you and your team? Customer: It was really helpful actually. We've been struggling with our current solution and your platform seems to address a lot of our pain points. Agent: That's great to hear! What specific challenges are you facing with your current setup? Customer: Well, the main issue is that our team is spending too much time on manual data entry and we're losing deals because we can't track follow-ups effectively. Agent: I understand. Our automation features could save your team about 15 hours per week on data entry alone. Plus, our smart follow-up system ensures no leads fall through the cracks. Customer: That sounds exactly like what we need. What's the next step? Agent: I'd love to set up a pilot program with your team. We can start with a 30-day trial at no cost, and you'll see the ROI within the first two weeks. Customer: That sounds perfect. Let's get started.",
            'conversion_label': 1,
            'duration_minutes': 12,
            'deal_size': 50000,
            'industry': 'Technology',
            'customer_type': 'Enterprise',
            'agent_experience_level': 'Senior'
        },
        {
            'transcript_id': 2,
            'call_transcript': "Agent: Hello Sarah, this is Mike from TechCorp. I'm calling about the email I sent regarding our new analytics solution. Customer: Oh yes, I saw that. We're actually pretty happy with our current analytics platform. Agent: I understand. What specific metrics are you tracking with your current solution? Customer: We track basic KPIs like conversion rates, customer acquisition costs, and revenue per customer. Agent: That's a solid foundation. Are you also tracking customer lifetime value, churn prediction, and real-time performance alerts? Customer: Not really. We mostly look at historical data. Agent: Our platform provides predictive analytics that can help you identify at-risk customers before they churn and optimize your marketing spend in real-time. Customer: That does sound valuable, but we're in the middle of our budget cycle. Maybe next quarter? Agent: I completely understand. Would you be open to a quick 15-minute demo to see how it works? No pressure, just to keep us in mind for when you're ready. Customer: Sure, I can spare 15 minutes. Agent: Perfect! I'll send you a calendar invite for tomorrow. Customer: Sounds good.",
            'conversion_label': 0,
            'duration_minutes': 8,
            'deal_size': 25000,
            'industry': 'Finance',
            'customer_type': 'SMB',
            'agent_experience_level': 'Mid-level'
        }
    ]
    
    import pandas as pd
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_transcripts.csv', index=False)
    print("✅ Sample data created successfully")
    return True

def create_contrastive_dataset():
    """Create contrastive dataset for training"""
    print("🎯 Creating contrastive dataset...")
    
    # Check if training script exists
    if not os.path.exists('training/contrastive_dataset.py'):
        print("❌ Contrastive dataset script not found")
        return False
    
    # Run the script
    return run_command("cd training && python contrastive_dataset.py", "Creating contrastive dataset")

def setup_environment():
    """Set up environment variables and configuration"""
    print("⚙️ Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = '.env'
    if not os.path.exists(env_file):
        env_content = """# Sales Conversion AI Environment Variables

# Model Configuration
FINETUNED_MODEL_PATH=models/finetuned_sales_model
GENERIC_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Training Parameters
LEARNING_RATE=2e-5
BATCH_SIZE=16
EPOCHS=10
WARMUP_STEPS=1000

# Evaluation
EVAL_METRICS=accuracy,precision,recall,f1_score,auc_roc
EVAL_OUTPUT_DIR=evaluation_results

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/sales_conversion.log

# UI Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Environment file created")
    
    # Create config.json
    config = {
        "model": {
            "generic_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 512,
            "embedding_dimension": 384
        },
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 10,
            "warmup_steps": 1000,
            "margin": 1.0
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
            "output_dir": "evaluation_results"
        },
        "ui": {
            "port": 8501,
            "host": "0.0.0.0"
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created")

def create_demo_script():
    """Create a demo script for quick testing"""
    demo_script = '''#!/usr/bin/env python3
"""
Demo script for Sales Conversion AI
Quick test of the system functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.embedder import SalesEmbedder
import pandas as pd

def main():
    print("🎯 Sales Conversion AI Demo")
    print("=" * 40)
    
    # Load sample data
    print("📊 Loading sample data...")
    df = pd.read_csv('data/sample_transcripts.csv')
    transcripts = df['call_transcript'].tolist()
    labels = df['conversion_label'].values
    
    # Initialize embedder (will use generic model for demo)
    print("🤖 Initializing embedder...")
    embedder = SalesEmbedder()
    
    # Compute prototypes
    print("🔧 Computing prototypes...")
    embedder.compute_prototypes(transcripts, labels)
    
    # Test prediction
    print("🔍 Testing prediction...")
    test_transcript = "Agent: Hi, I'm calling about your recent inquiry. Customer: Yes, we're interested in your solution. Agent: Great! Let me tell you about our features..."
    
    prediction = embedder.predict_conversion_score(test_transcript)
    
    print("\\n📈 Prediction Results:")
    print(f"Conversion Score: {prediction['conversion_score']:.3f}")
    print(f"Conversion Probability: {prediction['conversion_probability']:.1%}")
    print(f"Prediction: {'✅ Convert' if prediction['prediction'] else '❌ No Convert'}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    
    print("\\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
'''
    
    with open('demo.py', 'w') as f:
        f.write(demo_script)
    
    # Make it executable
    os.chmod('demo.py', 0o755)
    print("✅ Demo script created")

def create_launch_script():
    """Create launch script for the Streamlit app"""
    launch_script = '''#!/usr/bin/env python3
"""
Launch script for Sales Conversion AI Streamlit app
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Sales Conversion AI...")
    
    # Check if Streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Launch the app
    app_path = os.path.join("ui", "app.py")
    if os.path.exists(app_path):
        print("🌐 Starting Streamlit app...")
        print("📱 Access the app at: http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    else:
        print("❌ Streamlit app not found at ui/app.py")

if __name__ == "__main__":
    main()
'''
    
    with open('launch.py', 'w') as f:
        f.write(launch_script)
    
    # Make it executable
    os.chmod('launch.py', 0o755)
    print("✅ Launch script created")

def main():
    """Main setup function"""
    print("🎯 Sales Conversion AI Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download sample data
    if not download_sample_data():
        print("❌ Failed to create sample data")
        sys.exit(1)
    
    # Create contrastive dataset
    if not create_contrastive_dataset():
        print("⚠️ Failed to create contrastive dataset (this is optional)")
    
    # Setup environment
    setup_environment()
    
    # Create demo and launch scripts
    create_demo_script()
    create_launch_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run demo: python demo.py")
    print("2. Launch UI: python launch.py")
    print("3. Train model: cd training && python train_finetune.py")
    print("4. Evaluate: cd training && python evaluation.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 