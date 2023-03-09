# Generated by Django 4.1.7 on 2023-03-09 02:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('medicines', '0002_alter_medicine_is_etc'),
        ('users', '0002_remove_user_user_medicines_user_user_medicines'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='user_medicines',
            field=models.ManyToManyField(blank=True, related_name='users', to='medicines.medicine'),
        ),
    ]
