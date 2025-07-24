import country_converter as coco
cc = coco.CountryConverter()

def iso2_to_iso3(code):
    return cc.convert(code, to='ISO3')

def iso3_to_iso2(code):
    return cc.convert(code, to='ISO2')
