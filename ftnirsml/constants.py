#data standard and other values that will be referenced by ml-codebase functions and downstream products.

#order does not matter: internally, columns will be arranged into alphabetical order at time of model creation.
#for fine-tuning, if new columns are designated, will be appended to the right, in alphabetical order for any new colums.
#unselected columns will be blanked out .

#it really is pretty unimportant, what is important is that we rely a lot on the existence of metadata concerning the
#colum order for generated models. inference needs to know the order of the model object columns, and fine-tuning does
#as well (fine tuning will attempt to reorder the existing database equally, nullify no longer provided values,
#BACKFILLING new data columns, and if the net change was negative (fewer columns than what started with), provide a NULL
#name to signify this column was blanked out.

#here's how to do it: implement a masking layer that tells the model to ignore -1. DO NOT backfill, as the original
#weights may be useful later. Instead, if data are subtracted, just introduce the -1, and if new data are added,
#assign from the extra columns. Come up with a protocl to account for which columns were nullified, for posterity.

#steps: order new data alphabetically. Store metadata on columns, in order, as well as any nullified. Add a masking
#layer to the archetectures and make sure things are still working.

WN_MATCH = 'wavenumber_'
WN_STRING_NAME = f'{WN_MATCH}**'

IDNAME = "id"
SPLITNAME = "split"
RESPONSENAME = "age"

STANDARD_COLUMN_NAMES = {IDNAME:{'data_type':'unq_text'},
        SPLITNAME:{'data_type':'int'},
        'catch_year' : {'data_type':'int'},
        'catch_month': {'data_type': 'int'},
        'catch_day': {'data_type': 'int'},
        'sex' : {'data_type':'categorical'},
        RESPONSENAME : {'data_type':'numeric'},
        'latitude' : {'data_type':'numeric'},
        'longitude' : {'data_type':'numeric'},
        'region': {'data_type': 'categorical'},
        'fish_length': {'data_type': 'numeric'},
        'fish_weight': {'data_type': 'numeric'},
        'otolith_weight': {'data_type': 'categorical'},
        'gear_depth': {'data_type': 'numeric'},
        'gear_temp': {'data_type': 'numeric'}
        }

INFORMATIONAL = [IDNAME,SPLITNAME]
RESPONSE_COLUMNS = [RESPONSENAME]

#metadata fields that should be mandated by default for saving in keras.zip format after model train event:
#for the 'original' approaches, which includes basic model and hyperband tuning model
REQUIRED_METADATA_FIELDS_ORIGINAL = {"column_names","description","scaler_x","scaler_y"}

#define model approach metadata. Perhaps, model approach should be an object, and this should be
#attributes? Yeah, I think that is ok.
TRAINING_APPROACHES = {"Basic model":{'description':"basic, customizable model",'finetunable':True},
                       "hyperband tuning model":{'description':"A version of the basic model with hyperband parameter tuning",'finetunable':False}}


#other columns won't be defined, by definition, we will not know the appropriate datatype and instead will have logic to guess between
#categorical and continuous.

#define the input arguments of the different approaches
#not sure yet I need to do this, I can also extract the arguments from the functions themselves with:
#import inspect; inspect.signature(myfunction); for name,parm in signature.parameters.items():

#desire: easily extract necessary inputs in a reliable way from an object in the codebase.

#basic_model = {
#}