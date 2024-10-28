# census_transportation

### Install
```py
git clone https://github.com/zachkaupp/census_transportation.git

cd census_transportation

# REQUIRED: populate /data/raw

pip3 install -r requirements.txt

python3 -m census_transportation
```
### Data Files
- Use these files to populate /data/raw
- https://drive.google.com/drive/folders/1uQxTdbvacpcF2xq0uUvV_bXFjqfNhCZX?usp=sharing

### Project Notes

Goal: Find ways to practice pytorch with these datasets

Demographics:
https://catalog.data.gov/dataset/demographics-for-us-census-tracts-2012-american-community-survey-2008-2012-derived-summary-tabl8

Transportation:
https://catalog.data.gov/dataset/travel-time-to-work1
https://catalog.data.gov/dataset/means-of-transportation-to-work2

Life expectancy:
https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html#life-expectancy

#### Things to know:
- ID = unique code that matches to county by federal standards

### Future updates to increase accuracy
- Average life expectancy should be weighted by population for each census tract
- Take data strictly from the same years
- Investigate missing values instead of removing any features with missing values