# SOSense - Real-time Hand Gesture Recognition for Emergency Detection

SOSense is a computer vision application that detects emergency hand gestures in real-time and sends alerts via email and SMS.

## Setup Instructions

### 1. Environment Variables

Copy the example environment file and configure it with your credentials:

```bash
cp .env.example .env
```

Edit the `.env` file and replace the placeholder values with your actual credentials:

- **Email Configuration**: Set up Gmail app password for email alerts
- **Twilio Configuration**: Add your Twilio credentials for SMS alerts
- **Model Configuration**: Adjust detection thresholds if needed
- **File Paths**: Modify paths if you have a different project structure

### 2. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Train the Model (if needed)

If you need to retrain the model:

```bash
python training/train_hand_model.py
```

### 4. Run Real-time Detection

Start the real-time gesture detection:

```bash
python inference/real_time_detection.py
```

## Gesture Labels

The system recognizes the following gestures:
- `sfh` (Signal for Help) - Primary SOS gesture
- `need_help` - General help request
- `need_police` - Police assistance required
- `need_ambulance` - Medical emergency
- `visit_me` - Request for visit
- `call_me` - Request for call
- `not_yet_helped` - Still need assistance
- `i_am_okay` - All clear signal
- `neutral` - No specific gesture

## Security Notes

- Never commit your `.env` file to version control
- Use app-specific passwords for Gmail
- Keep your Twilio credentials secure
- Regularly rotate your API keys

## Project Structure

```
├── .env                    # Environment variables (not tracked)
├── .env.example           # Example environment file
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── datasets/             # Training data
├── inference/            # Real-time detection scripts
├── models/              # Trained models and scalers
├── script/              # Utility scripts
├── training/            # Model training scripts
└── utils/               # Feature extraction utilities
```
