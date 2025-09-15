"""
Celery app configuration
"""
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create Celery app instance
celery_app = Celery('chiller_optimization', broker=REDIS_URL, backend=REDIS_URL)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Include tasks from the tasks module
    include=['web.tasks']
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['web'])