# Quick Start Guide - ISL Telehealth Web App

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
cd webapp
pip install -r requirements_webapp.txt
```

### Step 2: Configure Doctor Emails (30 seconds)

Open `config.py` and add doctor email(s):

```python
DOCTOR_EMAILS = [
    'doctor@example.com',  # Replace with actual doctor email
]
```

### Step 3: Run the Application (30 seconds)

```bash
python app.py
```

You should see:
```
* Running on http://0.0.0.0:5000
```

### Step 4: Create Accounts (1 minute)

1. Open browser: `http://localhost:5000`
2. Click "Register"
3. Create a doctor account:
   - Name: Dr. Smith
   - Email: doctor@example.com (must be in DOCTOR_EMAILS)
   - Password: password123

4. Create a patient account:
   - Name: John Doe
   - Email: patient@example.com (any email NOT in DOCTOR_EMAILS)
   - Password: password123

### Step 5: Test the System (1 minute)

**Patient Side:**
1. Login as patient
2. Click "Start Detection"
3. Allow camera access
4. Perform sign language gestures

**Doctor Side:**
1. Open new browser tab/window
2. Login as doctor
3. View real-time translations
4. Send text responses

## ✅ Success Indicators

- ✅ Camera feed visible on patient side
- ✅ Signs detected and displayed
- ✅ Translations appear on doctor side
- ✅ Audio plays on doctor side
- ✅ Doctor messages reach patient

## 🐛 Common Issues

### Issue: "Model file not found"
**Solution**: Ensure you have trained the model first:
```bash
cd ../model_training
python train_model.py
```

### Issue: "Camera not accessible"
**Solution**: 
- Grant browser camera permissions
- Use HTTPS in production
- Try Chrome/Firefox

### Issue: "Module not found"
**Solution**: Install dependencies:
```bash
pip install -r requirements_webapp.txt
```

## 📝 Default Test Credentials

**Doctor:**
- Email: doctor@example.com
- Password: password123

**Patient:**
- Email: patient@example.com
- Password: password123

## 🎯 Next Steps

1. ✅ Test with real sign language gestures
2. ✅ Add more doctor emails in config.py
3. ✅ Customize UI in templates/
4. ✅ Deploy to production server

## 📞 Need Help?

Check the full README.md for detailed documentation.

---

**Happy Testing! 🤟**
