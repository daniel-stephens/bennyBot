from django.urls import path
from . import views
from django.contrib import admin

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('transcribe', views.transcribe, name='transcribe'),
    path("login/", views.login_view, name="login"),
    path('admin/', admin.site.urls),
    path("logout/", views.logout_view, name="logout"),
]
