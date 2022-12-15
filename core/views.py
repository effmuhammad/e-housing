from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import locale
from django.http import HttpResponse
from django.template.response import TemplateResponse
import json 
import csv
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import Profile, DataListrik, TestSimpan
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from pickle import load
from django.conf import settings
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

list_id_datalogger = [
    '7B83HH9T',
    'DSE5793M',
    'G7BDKATF',
    'HHG26A44',
    '6P67R3C6',
    'ESHK7LM8',
    'MS772733',
    '5695KABR',
    '6U2ZRSXT',
    'WURUDDRQ',
]

kwh_fee = 1352

def home(request):
    df = pd.read_csv("./telecom_users.csv")
    df = df[:10]
    # data = data.to_html()
    json_records = df.reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    contextt = {'d': arr}
    return render(request,'index.html',contextt)

def login(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        id_datalogger = request.POST['id_datalogger']
        email = request.POST['email']
        password = request.POST['password']

        if id_datalogger not in list_id_datalogger:
            messages.info(request, 'ID Datalogger Tidak Ditemukan')
            return redirect('login')

        user = auth.authenticate(id_datalogger = id_datalogger, username=email, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('dashboard')
        else:
            messages.info(request, 'Kredensial Tidak Tepat')
            return redirect('login')
    else:
        return render(request,'login.html')

def register(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        id_datalogger = request.POST['id_datalogger']
        namadepan = request.POST['namadepan']
        namabelakang = request.POST['namabelakang']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']

        if password == password2:
            if id_datalogger not in list_id_datalogger:
                messages.info(request, 'ID Datalogger Tidak Ditemukan')
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email Sudah Digunakan')
                return redirect('register')
            else: # jika benar
                user = User.objects.create_user(username=email, email=email, password=password)
                user.is_active = False
                user.first_name = namadepan
                user.last_name = namabelakang
                user.save()

                #log user in and redirect to settings page
                user_login = auth.authenticate(username=email, password=password)
                auth.login(request, user_login)
                
                return redirect('dashboard')
        else:
            messages.info(request, 'Password Tidak Cocok')
            return redirect('register')
    else:
        return render(request,'register.html')

@login_required(login_url='login')
def logout(request):
    auth.logout(request)
    return redirect('/')

@login_required(login_url='login')
def dashboard(request):
    df = load_data()
    now = datetime.datetime.now().date()
    start = now.strftime('%Y-%m-1')
    bulan = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    bulan_ini = bulan[now.month-1]
    # calc total penggunaan
    df['daya_semu'] = df['current'] * df['voltage']
    tot_use = format(sum(df['daya_semu'][start:now]/60)/1000, ".2f")
    # calc total tarif
    locale.setlocale(locale.LC_ALL, 'de_DE.utf-8')
    tot_tarif = locale.format('%d', int(float(tot_use) * kwh_fee), 1)
    # peak hour
    df_hour = df[-10080:].copy(deep=True)
    df_hour['hour'] = df_hour.index.hour
    peak_hour_val = df_hour.groupby('hour')['power'].mean().sort_values(ascending=False).index[0]
    peak_hour = datetime.time(peak_hour_val).strftime("%H:00")
    peak_hour_range = datetime.time(peak_hour_val+1).strftime("%H:00")

    # peak day
    df_day = df.copy(deep=True)
    df_day['day_name'] = df_day.index.day_name()
    peak_day_val = df_day[-43200:].groupby('day_name')['power'].mean().sort_values(ascending=False).index[0]
    day_id = ['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', "Jum'at", 'Sabtu']
    day_en = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    peak_day = day_id[day_en.index(peak_day_val)]

    # input date range

    # jika input kosong maka:
    end = '2022-8-20'
    start = pd.Timestamp(end) - pd.DateOffset(days=60)   # mundur 2 bulan

    time_step = 'D'
    df_pref = df[start:end].resample(time_step).mean().fillna(0)
    if time_step == 'D':
        pengali = 24
    elif time_step == 'H':
        pengali = 1
    elif time_step == 'T':
        pengali = 1/60
    df_pref['power_use']=df_pref['daya_semu']*pengali/1000
    df_pref['power_use_ch1']=df_pref['current_ch1']*df_pref['voltage']*pengali/1000_000
    df_pref['power_use_ch2']=df_pref['current_ch2']*df_pref['voltage']*pengali/1000_000
    df_pref['power_use_ch3']=df_pref['current_ch3']*df_pref['voltage']*pengali/1000_000


    # grafik power total
    fig_pow = px.line(
        x=df_pref.index.to_list(),
        y=df_pref['power_use'].to_list(),
        labels={'x':'Waktu', 'y':'Konsumsi Daya (kWh)'},
        template="plotly_white"
    )

    fig_pow.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        title={
            'font_size': 24,
            'xanchor': 'center',
            'x': 0.5
    })
    chart_pow = fig_pow.to_html()

    # persentase ch1, ch2, ch3
    mean_ch1 = df_pref['current_ch1'].mean()
    mean_ch2 = df_pref['current_ch2'].mean()
    mean_ch3 = df_pref['current_ch3'].mean()
    tot_mean = mean_ch1+mean_ch2+mean_ch3
    per_ch1=mean_ch1/tot_mean*100
    per_ch2=mean_ch2/tot_mean*100
    per_ch3=mean_ch3/tot_mean*100
    labels = ['Channel 1','Channel 2','Channel 3']
    values = [per_ch1, per_ch2, per_ch3]
    fig_per = go.Figure(data=[go.Pie(labels=labels, values=values, hole = 0.4, textinfo='label+percent')])
    fig_per.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        showlegend = False,
    )
    pie_per = fig_per.to_html()

    # grafik tegangan
    fig_volt = px.line(
        x=df_pref.index.to_list(),
        y=df_pref['voltage'].to_list(),
        labels={'x':'Waktu', 'y':'Tegangan (Volt)'},
        template="plotly_white"
    )

    fig_volt.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        title={
            'font_size': 24,
            'xanchor': 'center',
            'x': 0.5
    })
    chart_volt = fig_volt.to_html()

    # grafik power per channel
    fig_pow_ch = go.Figure()
    fig_pow_ch.add_trace(go.Scatter(x=df_pref.index.to_list(), y=df_pref['power_use_ch1'].to_list(), name='Channel 1',
                            line=dict(color='firebrick', width=2)))
    fig_pow_ch.add_trace(go.Scatter(x=df_pref.index.to_list(), y=df_pref['power_use_ch2'].to_list(), name='Channel 2',
                            line=dict(color='royalblue', width=2)))
    fig_pow_ch.add_trace(go.Scatter(x=df_pref.index.to_list(), y=df_pref['power_use_ch3'].to_list(), name='Channel 3',
                            line=dict(color='green', width=2)))
    fig_pow.update_yaxes(rangemode="tozero")
    fig_pow_ch.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        title={
            'font_size': 24,
            'xanchor': 'center',
            'x': 0.5
        },
        legend=dict(
            orientation="h",
            traceorder='normal',
            yanchor="top",
            y=1,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        template="plotly_white",
        xaxis_title='Waktu',
        yaxis_title='Konsumsi Daya (kWh)'
    )
    chart_pow_ch = fig_pow_ch.to_html()


    context = {'tot_use' : tot_use,
               'tot_tarif' : tot_tarif,
               'bulan_ini' : bulan_ini,
               'peak_hour' : peak_hour,
               'peak_hour_range' : peak_hour_range,
               'peak_day' : peak_day,
               'chart_pow' : chart_pow,
               'pie_per' : pie_per,
               'chart_volt' : chart_volt,
               'chart_pow_ch' : chart_pow_ch,
              }
    return render(request,'dashboard.html', context)

@login_required(login_url='login')
def prediksi(request):
    df = load_data()
    locale.setlocale(locale.LC_ALL, 'de_DE.utf-8')
    date_now = '2022-11-11' # datetime.datetime.now().date()
    window_size = 24
    df_hour = df.resample('H').mean().fillna(0)
    df_hour['day'] = df_hour.index.dayofweek 
    df_hour['hour'] = df_hour.index.hour
    df_hour['date'] = df_hour.index.date
    df_day = df.resample('D').mean().fillna(0)
    df_day = df.resample('D').mean().fillna(0)
    df_day['day'] = df_day.index.dayofweek
    df_day['date'] = df_day.index.date
    df_hour['daya_semu'] = df_hour['current'] * df_hour['voltage']
    # setiap jam
    per_jam = round(df_hour['daya_semu'][-168:].mean()/1000, 3)
    tarif_per_jam = locale.format('%d', int(float(per_jam) * kwh_fee), 1)
    # setiap hari
    per_hari = round(per_jam*24, 3)
    tarif_per_hari = locale.format('%d', int(float(per_hari) * kwh_fee), 1)
    # setiap bulan
    per_bulan = round(per_hari*30, 2)
    tarif_per_bulan = locale.format('%d', int(float(per_bulan) * kwh_fee), 1)
    # setiap tahun
    per_tahun = round(per_bulan*12, 2)
    tarif_per_tahun = locale.format('%d', int(float(per_tahun) * kwh_fee), 1)

    # grafik prediksi penggunaan global
    x_hour = ['00.00', '01.00', '02.00', '03.00', '04.00', '05.00', '06.00', '07.00', '08.00', '09.00', '10.00', '11.00', '12.00', '13.00', '14.00', '15.00', '16.00', '17.00', '18.00', '19.00', '20.00', '21.00', '22.00', '23.00']
    df_hour['power_use'] = df_hour['power']/1000

    y_data = df_hour['power_use'][date_now:'2022-11-11 18:00:00']
    y_data = list(y_data)+[None for i in range(24-len(y_data))]

    model = load_model('static/models/CNN-LSTM_power_model.h5')

    scaled = pd.DataFrame(0, index=np.arange(len(df_hour)), columns=['power'])
    scaler = load(open('static/models/scaler_power.pkl', 'rb'))
    scaled[['power']] = scaler.transform(df_hour[['power']])

    segment_temp = segment_multi(scaled, window_size, 1)
    pred = model.predict(segment_temp)

    pred = scaler.inverse_transform(pred)[0][0]/1000

    # pred = 0.123
    first_none = [enum for enum,i in enumerate(y_data) if i==None][0]
    y_pred_data = [None for i in range(first_none-1)]+[y_data[first_none-1]]+[pred]+[None for i in range(24-19)]

    # Create traces
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x_hour, y=y_minggu,
    #                 mode='markers',
    #                 name=str(date)))

    fig.add_trace(go.Scatter(x=x_hour, y=y_pred_data,
                    mode='lines+markers',line_color="#0cc70c",
                    name='Prediksi Konsumsi',
                    opacity = 1.0))
                    
    fig.add_trace(go.Scatter(x=x_hour, y=y_data,
                    mode='lines+markers',line_color="#315BBC",
                    name='Konsumsi Daya'))


    fig.update_xaxes(
            tickmode='linear',
            title_text = "Konsumsi Daya (kWh) terhadap Waktu", # label
            # title_font = {"size": 20},
            title_standoff = 10)

    fig.update_yaxes(rangemode="tozero")

    # set y axis label
    # fig.update_yaxes(
    #         title_text = "power (W)", # label
    #         title_standoff = 5)

    # fig.update_traces()
    fig.update_layout(
        # autosize=False, 
        height=300,
        margin=dict(
            l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )
    chart_pred_pow = fig.to_html()

    # grafik prediksi penggunaan per channel
    df_hour['power_use_ch1'] = df_hour['current_ch1']*df_hour['voltage']/1000
    df_hour['power_use_ch2'] = df_hour['current_ch2']*df_hour['voltage']/1000
    df_hour['power_use_ch3'] = df_hour['current_ch3']*df_hour['voltage']/1000
    y_data1 = df_hour['power_use_ch1'][date_now:'2022-11-11 18:00:00']
    y_data2 = df_hour['power_use_ch2'][date_now:'2022-11-11 18:00:00']
    y_data3 = df_hour['power_use_ch3'][date_now:'2022-11-11 18:00:00']
    y_data1 = list(y_data1)+[None for i in range(24-len(y_data1))]
    y_data2 = list(y_data2)+[None for i in range(24-len(y_data2))]
    y_data3 = list(y_data3)+[None for i in range(24-len(y_data3))]

    model1 = load_model('static/models/CNN-LSTM_current_ch1_model.h5')
    model2 = load_model('static/models/CNN-LSTM_current_ch2_model.h5')
    model3 = load_model('static/models/CNN-LSTM_current_ch3_model.h5')

    scaled1 = pd.DataFrame(0, index=np.arange(len(df_hour)), columns=['current_ch1'])
    scaled2 = pd.DataFrame(0, index=np.arange(len(df_hour)), columns=['current_ch2'])
    scaled3 = pd.DataFrame(0, index=np.arange(len(df_hour)), columns=['current_ch3'])
    scaler1 = load(open('static/models/scaler_current_ch1.pkl', 'rb'))
    scaler2 = load(open('static/models/scaler_current_ch2.pkl', 'rb'))
    scaler3 = load(open('static/models/scaler_current_ch3.pkl', 'rb'))
    scaled1[['current_ch1']] = scaler1.transform(df_hour[['current_ch1']])
    scaled2[['current_ch2']] = scaler2.transform(df_hour[['current_ch2']])
    scaled3[['current_ch3']] = scaler3.transform(df_hour[['current_ch3']])

    segment_temp1 = segment_multi(scaled1, window_size, 1)
    segment_temp2 = segment_multi(scaled2, window_size, 1)
    segment_temp3 = segment_multi(scaled3, window_size, 1)
    pred1 = model1.predict(segment_temp1)
    pred2 = model2.predict(segment_temp2)
    pred3 = model3.predict(segment_temp3)

    pred1 = scaler1.inverse_transform(pred1)[0][0]*220/1000
    pred2 = scaler2.inverse_transform(pred2)[0][0]*220/1000
    pred3 = scaler3.inverse_transform(pred3)[0][0]*220/1000

    first_none1 = [enum for enum,i in enumerate(y_data1) if i==None][0]
    first_none2 = [enum for enum,i in enumerate(y_data2) if i==None][0]
    first_none3 = [enum for enum,i in enumerate(y_data3) if i==None][0]
    y_pred_data1 = [None for i in range(first_none1-1)]+[y_data1[first_none1-1]]+[pred1]+[None for i in range(24-19)]
    y_pred_data2 = [None for i in range(first_none2-1)]+[y_data2[first_none2-1]]+[pred2]+[None for i in range(24-19)]
    y_pred_data3 = [None for i in range(first_none3-1)]+[y_data3[first_none3-1]]+[pred3]+[None for i in range(24-19)]

    # Create traces
    fig = go.Figure()

    # plot ch1
    fig.add_trace(go.Scatter(x=x_hour, y=y_pred_data1,
                    mode='lines+markers',line_color="#10A19D",
                    name='Prediksi Konsumsi CH1',
                    opacity = 0.4))
                    
    fig.add_trace(go.Scatter(x=x_hour, y=y_data1,
                    mode='lines+markers',line_color="#10A19D",
                    name='Konsumsi Daya CH1'))
    # plot ch2
    fig.add_trace(go.Scatter(x=x_hour, y=y_pred_data2,
                    mode='lines+markers',line_color="#540375",
                    name='Prediksi Konsumsi CH2',
                    opacity = 0.4))
                    
    fig.add_trace(go.Scatter(x=x_hour, y=y_data2,
                    mode='lines+markers',line_color="#540375",
                    name='Konsumsi Daya CH2'))
    # plot ch3
    fig.add_trace(go.Scatter(x=x_hour, y=y_pred_data3,
                    mode='lines+markers',line_color="#FF7000",
                    name='Prediksi Konsumsi CH3',
                    opacity = 0.4))
                    
    fig.add_trace(go.Scatter(x=x_hour, y=y_data3,
                    mode='lines+markers',line_color="#FF7000",
                    name='Konsumsi Daya CH3'))

    fig.update_xaxes(
            tickmode='linear',
            title_text = "Konsumsi Daya (kWh) terhadap Waktu", # label
            # title_font = {"size": 20},
            title_standoff = 10)

    fig.update_yaxes(rangemode="tozero")

    # set y axis label
    # fig.update_yaxes(
    #         title_text = "power (W)", # label
    #         title_standoff = 5)

    # fig.update_traces()
    fig.update_layout(
        # autosize=False, 
        height=300,
        margin=dict(
            l=0, r=0, b=0, t=0, pad=0),
        # legend=dict(
        #     orientation="h",
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01,
        # ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    chart_pred_ch = fig.to_html()

    # grafik trend penggunaan
    df_day['power_use'] = df_day['power']*24/1000
    Y = df_day['power_use']
    X = np.arange(1,len(df_day)+1)
    reg = LinearRegression().fit(np.vstack(X), Y)
    df_day['regression'] = reg.predict(np.vstack(X))

    fig = px.histogram(df_day, 
        x=df_day.index, 
        y="power_use", histfunc="avg", 
        # labels = {
        #     'x' : 'ABD',
        #     'y' : 'Time'
        # }
    )
    fig.update_traces(xbins_size="M1")
    fig.update_xaxes(showgrid=True, ticklabelmode="period", dtick="M1", tickformat="%b\n%Y")
    fig.update_xaxes(title_text='Bulan')
    fig.update_yaxes(title_text='Konsumsi Daya (kWh)')
    fig.update_layout(bargap=0.1)
    fig.add_trace(go.Scatter(mode="markers", x=df_day.index, y=df_day["power_use"], name="Penggunaan Harian"))
    fig.add_trace(go.Scatter(mode="lines", x=df_day.index, y=df_day["regression"], name="Garis Trend"))
    # fig.update_xaxes(dtick="M1",tickformat="%d %B %Y")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        template="plotly_white",
    )
    chart_reg = fig.to_html()
    
    # grafik pola penggunaan harian
    hari = 6    # 0 Senin, 6 Minggu
    jml_minggu = 10

    i = 0
    for index, row in df_hour[:-24:-1].iterrows():
        if row['hour'] == 23:
            break
        i+=1
    if i == 0: stepback = None
    else: stepback = -i
    df_hour_seg = df_hour[-(168*jml_minggu)-i:stepback]
    df_hour_seg['power_use'] = df_hour_seg['power']/1000
    x_hour = ['00.00', '01.00', '02.00', '03.00', '04.00', '05.00', '06.00', '07.00', '08.00', '09.00', '10.00', '11.00', '12.00', '13.00', '14.00', '15.00', '16.00', '17.00', '18.00', '19.00', '20.00', '21.00', '22.00', '23.00']
    # Create traces
    fig = go.Figure()
    for date in df_hour_seg.loc[df_hour_seg['day']==hari]['date'].unique():
        y_minggu = df_hour_seg['power_use'].loc[df_hour_seg['date']==date]
        fig.add_trace(go.Scatter(x=x_hour, y=y_minggu,
                        mode='markers',
                        name=str(date)))
    y_mean = df_hour_seg.loc[df_hour_seg['day']==hari].groupby('hour')['power_use'].mean()       
    fig.add_trace(go.Scatter(x=x_hour, y=y_mean,
                    mode='lines',
                    name='Rata-rata'))
    fig.update_xaxes(
            title_text = "Konsumsi Daya (kWh) terhadap Waktu", # label
            # title_font = {"size": 20},
            title_standoff = 10)
    # set y axis label
    # fig.update_yaxes(
    #         title_text = "power (W)", # label
    #         title_standoff = 5)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        # paper_bgcolor="LightSteelBlue",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    chart_pat_D = fig.to_html()

    # grafik pola penggunaan mingguan
    jml_minggu = 12
    i = 0
    for index, row in df_day[:-8:-1].iterrows():
        if row['day'] == 6:
            break
        i+=1
    if i == 0: stepback = None
    else: stepback = -i
    df_day_seg = df_day[-(7*jml_minggu)-i:stepback]
    df_day_seg['power_use'] = df_day_seg['power']*24/1000
    list_minggu = []
    for i in range(1,jml_minggu+1):
        for j in range(7): list_minggu.append(i)
    df_day_seg['minggu'] = list_minggu
    x_day = ['Senin', 'Selasa', 'Rabu', 'Kamis', "Jum'at", 'Sabtu', 'Minggu'] # ind
    # x_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', "Friday", 'Saturday', 'Sunday'] # ing
    # Create traces
    fig = go.Figure()
    for minggu in df_day_seg['minggu'].unique():
        y_minggu = df_day_seg['power_use'].loc[df_day_seg['minggu']==minggu]
        fig.add_trace(go.Scatter(x=x_day, y=y_minggu,
                        mode='markers',
                        name='Minggu ke-'+str(minggu)))
    y_mean = df_day_seg.groupby('day')['power_use'].mean()       
    fig.add_trace(go.Scatter(x=x_day, y=y_mean,
                    mode='lines',
                    name='Rata-rata'))
    fig.update_xaxes(
            title_text = "Konsumsi Daya (kWh) terhadap Hari", # label
            # title_font = {"size": 20},
            title_standoff = 8)
    # set y axis label
    # fig.update_yaxes(
    #         title_text = "power (W)", # label
    #         title_standoff = 5)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        height=350,
        margin=dict(
            l=0, r=0, b=0, t=0, pad=0),
        # paper_bgcolor="LightSteelBlue",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    chart_pat_W = fig.to_html()

    context = {'per_jam' : per_jam,
               'per_hari' : per_hari,
               'per_bulan' : per_bulan,
               'per_tahun' : per_tahun,
               'tarif_per_jam' : tarif_per_jam,
               'tarif_per_hari' : tarif_per_hari,
               'tarif_per_bulan' : tarif_per_bulan,
               'tarif_per_tahun' : tarif_per_tahun,
               'chart_pred_pow' : chart_pred_pow,
               'chart_pred_ch' : chart_pred_ch,
               'chart_reg' : chart_reg,
               'chart_pat_D' : chart_pat_D,
               'chart_pat_W' : chart_pat_W,
              }
    return render(request,'analytics-prediksi.html', context)

@login_required(login_url='login')
def anomali(request):
    context = {}
    return render(request,'analytics-anomali.html', context)

def forgot(request):
    return render(request,'forgot-password.html')

@csrf_exempt
def post_example_raw(request):
    print('POST', request.POST)
    print('BODY', request.body)
    print('JSON', json.loads(request.body.decode('utf-8')))
    return HttpResponse("200")

@csrf_exempt
def test(request):
    data = json.loads(request.body.decode('utf-8'))
    print('JSON', data)
    print(data['datetime'])
    # id_datalogger = data['id_datalogger'],
    try:
        TestSimpan.objects.create(
            datetime = data['datetime'],
            data1 = data['data1'],
            data2 = data['data2'],
        )
    except:
        print('error')

    return HttpResponse(200)

@login_required(login_url='login')
def downloaddata(request):
    # load_data()
    response = HttpResponse(content_type='text/csv')  
    response['Content-Disposition'] = 'attachment; filename="e-housing_data.csv"'  
    # writer = csv.writer(response)
    # writer.writerow(['1001', 'John', 'Domil', 'CA'])  
    # writer.writerow(['1002', 'Amit', 'Mukharji', 'LA', '"Testing"'])  
    load_data().to_csv(response, index = None, header=True)
    return response

def pandas(request):
    query = str(TestSimpan.objects.all().query)
    df = pd.read_sql_query(query, connection)
    print(df)
    return HttpResponse(df)

def segment_multi(data, window_size, col_size):
    segments = np.empty((0,window_size,col_size))
    labels = np.empty((0))
    list_data = []
    for col in data.columns:
        list_data.append(data[col][-window_size:])
        segments = np.vstack([segments,np.dstack(list_data)])
    return segments

def load_data():
    df = pd.DataFrame()
    paths = ['static/dataset/09-07-2022_00.00_sampai_11-11-2022_23.59.csv',
            ]
    for path in paths:
        df_read = pd.read_csv(path, sep=',', 
            names=['datetime','voltage','current','power','frequency','power_factor','current_ch1','current_ch2','current_ch3'],
            parse_dates=['datetime'],
            infer_datetime_format=True,
            index_col='datetime',
            low_memory=False, na_values=['nan'])
        df = pd.concat([df, df_read])

    df = df.resample('T').mean().fillna(0)
    
    # load data dari database
    # query = str(TestSimpan.objects.all().query)
    # df_read = pd.read_sql_query(query, connection
    #     parse_dates=['datetime'],
    #     infer_datetime_format=True,
    #     index_col='datetime')
    # df = pd.concat([df, df_read])

    return df