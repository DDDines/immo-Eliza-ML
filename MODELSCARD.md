# Model card

## Project context

This project involves the development of an AI model to predict prices using polynomial regression. The process encompasses data cleaning, outlier management, polynomial transformations, and the visualization of results to assess model performance.

## Data

dataset = data\properties.csv

target variable = Price

features = nbr_frontages,nbr_bedrooms,latitude,longitude,total_area_sqm,surface_land_sqm,terrace_sqm,garden_sqm,construction_year",primary_energy_consumption_sqm,cadastral_income,fl_furnished,fl_open_fire,fl_terrace,fl_garden,fl_swimming_pool,fl_floodzone,fl_double_glazing

## Model details

Started with linear regression, it was a decent start but not quite what I needed.Then, switched to polynomial regression, which really seemed to click with what I was aiming for, based on what I learned while experimenting with it.

## Performance

Train R² score: 0.6440179791111824
Test R² score: 0.6343437057122793

## Limitations

The main challenge with our approach is that it might get a bit too eager, trying to fit every tiny detail, even the ones that don't matter much. This means it make things more complicated than they need to be, especially when it's trying to guess new prices it hasn't seen before.

## Usage

What are the dependencies, what scripts are there to train the model, how to generate predictions, ...

## Maintainers

Contact: Julio Cesar Barizon Montes
e-Mail: juliobarizon@gmail.com
