{% extends 'markdown/index.md.j2'%}

{% block input %}
{%- set text = cell.source| replace("\nshow()", "") %}
{%- if text %}
```julia
{%- if text.endswith(";") %}
{{ text[:-1] }}
{%- else %}
{{ text }}
{%- endif %}
```
{% endif %}
{% endblock input %}

{%- block markdowncell %}
{%- set parts = cell.source.split("$$") %}
{%- for i in range(parts | count) %}
{%- if i % 2 %}
```math
{{ parts[i] }}
```
{%- else %}{{ parts[i] | replace("$", "``") }}{%- endif %}
{%- endfor %}
{% endblock markdowncell %}
