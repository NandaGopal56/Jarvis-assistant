# forms.py
from django import forms
from .models import User, GenderChoices, RoleChoices

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name', 'gender', 'mobile', 'country', 'profile_picture']

    # Make email read-only
    email = forms.EmailField(disabled=True)
    gender = forms.ChoiceField(choices=GenderChoices.choices())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs.update({
                'class': 'bg-gray-800 text-white rounded-lg px-4 py-2'
            })