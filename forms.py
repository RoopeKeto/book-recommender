from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange

class SearchForm(FlaskForm):
    searchword = StringField('Search word',
                             validators=[DataRequired(), Length(min=1, max=200)],
                             
                             )
    search = SubmitField('Search')