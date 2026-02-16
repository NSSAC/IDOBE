# Raw data
This folder contains the curated outbreaks time series data `outbreaks_disease_location.csv`. \
The dataset consists of curated time series for 13 different diseases, across 248 unique locations, and outcomes such as outpatient visits, confirmed cases and hospitalizations. The dataset comprising 10799 outbreaks, is compiled from existing disease data repositories such as Tycho, JHU-CSSE COVID-19 data repository, as well public health surveillance published by US CDC and the National Healthcare Safety Network (NHSN).\

The data dictionary is as follows:

    unique_id -- An unique identification number for each outbreak
    disease -- Type of disease (COVID-19, influenza, Smallpox, etc.)
    location -- Name of the location (US states, countries, etc.)
    event -- Type of burden indicator (cases, hospitalizations, etc.)
    start_date -- Start date of the outbreak (end-of-week Saturday date)
    end_date -- End date of the outbreak (end-of-week Saturday date)
    duration -- Duration of the outbreak (in weeks)
    [0-59] -- Values observed for the particular outbreak for given week (counts, %) 

