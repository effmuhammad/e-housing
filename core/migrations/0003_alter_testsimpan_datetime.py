# Generated by Django 3.2.14 on 2022-10-12 10:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_auto_20221012_1702'),
    ]

    operations = [
        migrations.AlterField(
            model_name='testsimpan',
            name='datetime',
            field=models.CharField(max_length=20),
        ),
    ]
