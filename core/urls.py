from django.urls import path
from django.contrib.auth import views as auth_views

from . import views
from . import models

urlpatterns = [
    path('',views.home, name = 'home'),
    path('', auth_views.LoginView.as_view(redirect_authenticated_user = True), name = 'login'),
    path('login',views.login, name = 'login'),
    path('register',views.register, name = 'register'),
    path('dashboard',views.dashboard, name = 'dashboard'),
    path('forgot-password',views.forgot, name = 'forgot-password'),
    path('logout', views.logout, name='logout'),
    path('post_example_raw', views.post_example_raw, name='post_example_raw'),
    path('test', views.test, name='test'),
    path('downloaddata', views.downloaddata, name='downloaddata'),
    path('pandas', views.pandas, name='pandas'),
    path('prediksi', views.prediksi, name='prediksi'),
    path('anomali', views.anomali, name='anomali'),
]