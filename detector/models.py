from django.db import models

class FireEvent(models.Model):
    detected_at = models.DateTimeField(auto_now_add=True)
    duration = models.DurationField()

    def __str__(self):
        return f"Fire at {self.detected_at} for {self.duration}"
