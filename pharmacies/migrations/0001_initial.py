# Generated by Django 4.1.7 on 2023-03-08 07:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Pharmacy',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100, verbose_name='약국명')),
                ('callNumber', models.CharField(max_length=20, null=True, verbose_name='연락처')),
                ('address', models.CharField(max_length=140, verbose_name='주소')),
                ('coordinate_X', models.CharField(max_length=40)),
                ('coordinate_Y', models.CharField(max_length=40)),
            ],
        ),
    ]
