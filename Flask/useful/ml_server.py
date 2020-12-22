#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
import ensembles as es


# In[14]:


import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField
from wtforms import SelectField, IntegerField, FileField, TextField, FloatField


# In[ ]:

PEOPLE_FOLDER = os.path.join('pictures')
app = Flask(__name__, template_folder='html', static_folder="pictures")
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
models = {}
csv_nums = 1
messages = []
info_about_models = {}


class Message:
    header = ''
    text = ''
    accur = 'accuracy is not available when test_part=0'

class data_model:
    alg = ''
    train = ''
    targ = ''

class TextForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    submit = SubmitField('Get Result')

class model_params(FlaskForm):
    select = SelectField('Select model type', choices=['RandomForest', 'GradientBoosting'], validators=[DataRequired()])
    train = FileField('Choose the file with train set data', validators=[DataRequired()])
    targ = FileField('Choose the file with train set target', validators=[DataRequired()])
    text = StringField('Model`s name', validators=[DataRequired()])
    siz = FloatField('Optional: test part(0.0-1.0)', default=0.2)
    n_estimators = IntegerField('n_estimators(recomendation for GradientBoosting:250, recomendation for RandomForest:500)', validators=[DataRequired()], default=250)
    max_depth = IntegerField('max_depth for each tree(recomendation for GradientBoosting:5, recomendation for RandomForest:15)', validators=[DataRequired()], default=5)
    fss = FloatField('feature subsample size(recomendation for GradientBoosting:0.5, recomendation for RandomForest:1.0)',  validators=[DataRequired()], default=0.5)
    learning_rate = FloatField('learning rate(only for GradientBoosting, recomendation 0.1)', validators=[DataRequired()], default=0.1)
    submit = SubmitField('Fit model!')



class for_predict(FlaskForm):
    test = FileField('Choose the file with set for prediction', validators=[DataRequired()])
    name = StringField('Input the name of file  with predictions', default='predictions_1.csv')
    #csv_nums += 1
    #name = StringField('Input the name of file  with predictions', default='predictions_{}.csv'.format(csv_nums))
    submit = SubmitField('Predict!')

class init_page(FlaskForm):
    name = StringField('Input model`s name', default='')
    submit = SubmitField('Predict setting')
    submit_create = SubmitField('Create new model')


class Response(FlaskForm):
    score = StringField('Score', validators=[DataRequired()])
    sentiment = StringField('Sentiment', validators=[DataRequired()])
    submit = SubmitField('Try Again')


def analyz_nan(data):
    data = data.drop(columns=['index', 'id', 'date'])
    MISS = data.isna().sum()[data.isna().sum()!=0].index
    for i in MISS:
        data[i] = data[i].fillna(data[i].mean(skipna=True))
        #print(data[i].mean(skipna=True))
    return data


def rand_for(data_name_train, data_name_targ,val=None, **kwargs):
    try:
        train = pd.read_csv('data/'+data_name_train)
        targ = pd.read_csv('data/'+data_name_targ)
        val_train, val_targ = None, None
        train = analyz_nan(train)
        targ = targ.drop(columns=['index'])
        if not val is None:
            train, val_train, targ, val_targ = train_test_split(train, targ, test_size=val)
            val_train = val_train.to_numpy()
            val_targ = val_targ.to_numpy().T[0]
        del kwargs['learning_rate']
        kwargs['feature_subsample_size'] = int(kwargs['feature_subsample_size']*train.shape[1])
        alg = es.RandomForestMSE(**kwargs)
        his = alg.fit(train.to_numpy(), targ.to_numpy().T[0], val_train, val_targ)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        #score, sentiment = 0.0, 'unknown'

    return alg, his

def makegraph(accurac, name):
    if not accurac is None:
        plt.figure(figsize=(9,5))
        plt.plot(range(len(accurac)), accurac)
        plt.xlabel('number of trees')
        plt.ylabel('accuracy(RMSE)')
        plt.grid()
        plt.title('Dependence of accurasy on test part on the number of trees')
        plt.savefig('pictures/'+name+'.jpg')
    else:
        image = Image.open("pictures/not_exist.jpg")
        image.save('pictures/'+name+'.jpg')
        


def makegraph_loss(accurac, name):
    plt.figure(figsize=(9,5))
    plt.plot(range(len(accurac)), accurac)
    plt.xlabel('number of trees')
    plt.ylabel('loss(RMSE)')
    plt.grid()
    plt.title('Dependence of loss function on the number of trees')
    plt.savefig('pictures/'+name+'_loss'+'.jpg')

def grad_bus(data_name_train, data_name_targ, val=None, **kwargs):
    try:
        train = pd.read_csv('data/'+data_name_train)
        targ = pd.read_csv('data/'+data_name_targ)
        val_train, val_targ = None, None
        train = analyz_nan(train)
        targ = targ.drop(columns=['index'])
        if not val is None:
            train, val_train, targ, val_targ = train_test_split(train, targ, test_size=val)
            val_train = val_train.to_numpy()
            val_targ = val_targ.to_numpy().T[0]
        kwargs['feature_subsample_size'] = int(kwargs['feature_subsample_size']*train.shape[1])
        alg = es.GradientBoostingMSE( **kwargs)
        his = alg.fit(train.to_numpy(), targ.to_numpy().T[0], val_train, val_targ)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        #score, sentiment = 0.0, 'unknown'
    return alg, his

def convert_to_str(parametr):
    info = []
    for i in parametr:
        if i!='accur_graph':
            strin = i+' = {}'.format(parametr[i])
            info.append(strin)
    return info

def get_par(par_form):
    parametr = {}
    parametr['n_estimators'] = par_form.n_estimators.data
    parametr['max_depth'] = par_form.max_depth.data
    parametr['feature_subsample_size'] = par_form.fss.data
    parametr['learning_rate'] = par_form.learning_rate.data
    return parametr

@app.route('/')
@app.route('/index')
def index():
    for the_file in os.listdir(PEOPLE_FOLDER):
        file_path = os.path.join(PEOPLE_FOLDER, the_file)
        if os.path.isfile(file_path) and the_file!='not_exist.jpg':
            os.unlink(file_path)
    return redirect(url_for('choose_model'))


@app.route('/index_js')
def get_index():
    return '<html><center><script>document.write("Hello, i`am Flask Server!")</script></center></html>'


@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    messages.clear()
    return redirect(url_for('prepare_message'))


@app.route('/choozing_model', methods=['GET', 'POST'])
def choose_model():
    init_form = init_page()

    if init_form.validate_on_submit():
        print(init_form.submit.data)
        if init_form.submit_create.data:
            return redirect(url_for('fitting_model'))
        else:
            if init_form.name.data=='':
                return redirect(url_for('choose_model'))
            return redirect(url_for('predict_page', mod_name=init_form.name.data))

    return render_template('chosmod.html', form=init_form, messages=messages)


@app.route('/predict_page', methods=['GET', 'POST'])
def predict_page():
    try:
        pred_form = for_predict()
        #pred_form.name = StringField('Input the name of file  with predictions', default='predictions_{}.csv'.format(csv_nums))
        #app.logger.info('On text: {0}'.format(par_form.select.data))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], request.args.get('mod_name')+'.jpg')
        pic_loss = os.path.join(app.config['UPLOAD_FOLDER'], request.args.get('mod_name')+'_loss'+'.jpg')
        string = convert_to_str(info_about_models[request.args.get('mod_name')])
        print(full_filename)
        if pred_form.validate_on_submit():
            test = pd.read_csv('data/'+pred_form.test.data.filename)
            pr = models[request.args.get('mod_name')].predict(analyz_nan(test).to_numpy())
            df = pd.DataFrame(pr, columns=['predictions'], index=test.index)
            name = pred_form.name.data
            df.to_csv('data/'+name)
            return redirect(url_for('choose_model'))
        return render_template('predictions_page.html', form=pred_form, messages=string, picture_loss=pic_loss, picture=full_filename)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/model_train', methods=['GET', 'POST'])
def fitting_model():
    try:
        par_form = model_params()
        model = data_model
        if par_form.validate_on_submit():
            #app.logger.info('On text: {0}'.format(par_form.select.data))
            model.alg = par_form.select.data
            model.train = par_form.train.data.filename
            model.targ = par_form.targ.data.filename
            parametr = get_par(par_form)
            if par_form.siz.data:
                model.testsiz = par_form.siz.data
            else:
                model.testsiz = None
            if model.alg=='RandomForest':
                model, hist = rand_for(model.train, model.targ, model.testsiz, **parametr)
            else:
                model, hist = grad_bus(model.train, model.targ, model.testsiz, **parametr)
            parametr['train_set_name'] = par_form.train.data.filename
            parametr['model_type'] = par_form.select.data
            models[par_form.text.data] = model
            parametr['learning time(sec)'] = round(np.sum(np.array(hist[2])),2)
            makegraph(hist[1], par_form.text.data)
            makegraph_loss(hist[0], par_form.text.data)
            info_about_models[par_form.text.data] = parametr
            msg = Message()
            msg.header = 'model_{}'.format(len(messages))
            msg.text = par_form.text.data
            #print(hist)
            if par_form.siz.data:
                msg.accur = "accuracy(RMSE) = %.4f" % hist[1][-1]
            messages.append(msg)
            return redirect(url_for('choose_model'))
        return render_template('from_form.html', form=par_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
