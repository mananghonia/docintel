from rest_framework.routers import DefaultRouter

from training.views import ModelVersionViewSet, TrainingRunViewSet

router = DefaultRouter()
router.register("models", ModelVersionViewSet, basename="models")
router.register("runs", TrainingRunViewSet, basename="runs")

urlpatterns = router.urls
