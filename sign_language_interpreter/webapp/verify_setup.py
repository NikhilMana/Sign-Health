import sys
from pathlib import Path

print("=" * 60)
print("Sign Health Web App - Setup Verification")
print("=" * 60)

errors = []
warnings = []

# Check Python version
print("\n1. Checking Python version...")
if sys.version_info >= (3, 8):
    print("   ✓ Python version OK:", sys.version.split()[0])
else:
    errors.append("Python 3.8+ required")

# Check model files
print("\n2. Checking model files...")
model_dir = Path(__file__).parent.parent / 'models'
required_files = ['sign_model.keras', 'label_encoder.pkl', 'max_length.txt']

for file in required_files:
    if (model_dir / file).exists():
        print(f"   ✓ {file} found")
    else:
        errors.append(f"Missing: {file}")

# Check dependencies
print("\n3. Checking dependencies...")
required_modules = [
    'flask', 'flask_login', 'flask_socketio', 
    'flask_cors', 'bcrypt', 'gtts', 'cv2', 
    'mediapipe', 'tensorflow', 'numpy'
]

for module in required_modules:
    try:
        __import__(module)
        print(f"   ✓ {module}")
    except ImportError:
        errors.append(f"Missing module: {module}")

# Check directories
print("\n4. Checking directories...")
dirs = ['templates', 'static', 'static/css', 'static/js', 'models', 'services']
for d in dirs:
    if (Path(__file__).parent / d).exists():
        print(f"   ✓ {d}/")
    else:
        warnings.append(f"Missing directory: {d}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("❌ ERRORS FOUND:")
    for e in errors:
        print(f"   - {e}")
else:
    print("✅ ALL CHECKS PASSED!")

if warnings:
    print("\n⚠ WARNINGS:")
    for w in warnings:
        print(f"   - {w}")

print("\n" + "=" * 60)
if not errors:
    print("Ready to run: python app.py")
print("=" * 60)
