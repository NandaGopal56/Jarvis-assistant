from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='add_class')
def add_class(field, css_class):
    """
    Add custom CSS classes to Django form fields.
    """
    if hasattr(field, 'field'):
        # Add the CSS class to the widget's attributes
        existing_classes = field.field.widget.attrs.get('class', '')
        field.field.widget.attrs['class'] = f"{existing_classes} {css_class}".strip()
        return field
    else:
        # If field is already rendered, return it as is
        return mark_safe(field)
