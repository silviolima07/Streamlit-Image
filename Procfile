#web: gunicorn main:app --preload
web: sh setup.sh && streamlit run --server.enableCORS false --server.port $PORT opencv-image.py
