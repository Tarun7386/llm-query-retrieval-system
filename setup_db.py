import os
# Set environment variables before importing
os.environ['GEMINI_API_KEY'] = 'AIzaSyDntlATOYuzBmNRj1uktf87dQdYqiBohoo'
os.environ['PINECONE_API_KEY'] = 'pcsk_3DZqjX_D6FJgnKidV89NV73i4qznDAYp5kyzgQQbWu3Z9v5MfXGrSxjAESSuvWoPSNoUYz'

from database import engine
from models import database_models as m

print("Creating database tables...")
try:
    m.Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
