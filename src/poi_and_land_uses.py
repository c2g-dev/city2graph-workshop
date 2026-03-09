import os
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

os.makedirs(DATA_DIR, exist_ok=True)

CATEGORY_MAPPING = {
    # RETAIL: GENERAL GOODS
    'retail': [
        # General & Department
        'antique_store', 'business_office_supplies_and_stationery', 'charity_shop',
        'department_store', 'discount_store', 'duty_free_shop', 'flea_market',
        'gift_shop', 'kiosk', 'outlet_store', 'packing_supply', 'party_supply',
        'pawn_shop', 'pop_up_shop', 'rental_kiosks', 'retail', 'second_hand_store',
        'shopping', 'shopping_center', 'shopping_mall', 'souvenir_shop', 'superstore',
        'thrift_store', 'used_vintage_and_consignment', 'warehouse_store',

        # Fashion & Clothing
        'baby_gear_and_furniture', 'baby_store', 'boutique', 'bridal_shop', 'bridal_store',
        'childrens_clothing_store', 'clothing_store', 'costume_store', 'custom_t_shirt_store',
        'dance_wear', 'fashion', 'fashion_accessories_store', 'furrier', 'hat_shop',
        'jewelry_store', 'leather_goods_store', 'linen', 'luggage_store', 'maternity_wear',
        'mens_clothing_store', 'plus_size_fashion', 'shoe_store', 'sunglasses_store',
        'swimwear_store', 'traditional_clothing', 'vintage_clothing_store', 'watch_store',
        'wig_store', 'womens_clothing_store',

        # Home, Garden & Hardware
        'appliance_store', 'bedding_and_bath_stores', 'building_supply_store', 'carpet_store',
        'electrical_supply_store', 'fireplace_store', 'flooring', 'flowers_and_gifts_shop',
        'flower_shop', 'florist', 'framing_store', 'furniture_store', 'garden_center',
        'glass_and_mirror_shop', 'hardware_store', 'home_decor', 'home_goods_store',
        'home_improvement_store', 'kitchen_supplies', 'lighting_store', 'mattress_store',
        'nursery_and_gardening', 'paint_store', 'shades_and_blinds', 'wallpaper_store',
        'window_treatment_store',

        # Electronics & Photo
        'camera_store', 'computer_store', 'electronics', 'electronics_store',
        'mobile_phone_store', 'music_and_dvd_store', 'printing_equipment_and_supply',
        'telecommunications', 'video_game_store',

        # Hobbies, Music & Art
        'art_supplies', 'arts_and_crafts', 'bicycle_shop', 'bookshop', 'bookstore',
        'comic_book_store', 'dive_shop', 'fabric_store', 'golf_equipment', 'guitar_store',
        'gun_shop', 'hobby_shop', 'hunting_and_fishing_supplies', 'music_store',
        'musical_instrument_store', 'newspaper_and_magazines_store', 'office_supplies',
        'outdoor_supply_store', 'record_shop', 'sewing_supply', 'skate_shop',
        'ski_and_snowboard_shop', 'sporting_goods', 'sporting_goods_store',
        'stationery_store', 'surf_shop', 'toy_store', 'trophy_shop', 'vinyl_record_store',

        # Specialty, Beauty Products & Adult
        'adult_store', 'beauty_product_supplier', 'candle_store', 'cosmetic_and_beauty_supplies',
        'cosmetics_and_beauty_supply', 'e_cigarette_store', 'firework_retailer',
        'hair_supply_stores', 'hearing_aids', 'lottery_ticket', 'perfume_store',
        'pet_food', 'pet_store', 'pet_supplies', 'reptile_shop', 'tobacco_shop', 'vape_store',

        # Groceries & Markets
        'convenience_store', 'farmers_market', 'frozen_foods', 'fruit_and_vegetable_store',
        'grocery_store', 'greengrocer', 'health_food_store', 'health_market',
        'international_grocery_store', 'korean_grocery_store', 'market',
        'organic_grocery_store', 'public_market', 'specialty_grocery_store', 'supermarket',

        # Specialty Food
        'butcher', 'butcher_shop', 'cheese_shop', 'dairy_stores', 'fishmonger',
        'honey_farm_shop', 'meat_shop', 'specialty_foods',

        # Alcohol (Retail)
        'beer_store', 'beverage_store', 'liquor_store', 'wine_shop',

        # (added from unmapped_categories.txt)
        'academic_bookstore',
        'art_supply_store',
        'audio_visual_equipment_store',
        'beer_wine_and_spirits',
        'cards_and_stationery_store',
        'coffee_and_tea_supplies',
        'coin_dealers',
        'comic_books_store',
        'designer_clothing',
        'diamond_dealer',
        'educational_supply_store',
        'fitness_exercise_equipment',
        'fruits_and_vegetables',
        'furniture_accessory_store',
        'gun_and_ammo',
        'handbag_stores',
        'herb_and_spice_shop',
        'home_and_garden',
        'home_theater_systems_stores',
        'imported_food',
        'kitchen_and_bath',
        'kitchen_supply_store',
        'knitting_supply',
        'leather_goods',
        'lingerie_store',
        'meat_wholesaler',
        'motorsports_store',
        'outdoor_gear',
        'pen_store',
        'seafood_market',
        'sports_wear',
        'tabletop_games',
        'uniform_store',
        'used_bookstore',
        'vitamins_and_supplements',
        'wholesale_store',
        '3d_printing_service',
        'aquatic_pet_store',
        'auto_parts_and_supply_store',
        'awning_supplier',
        'bags_luggage_company',
        'boat_parts_and_accessories',
        'boat_parts_and_supply_store',
        'books_mags_music_and_video',
        'cabinet_sales_service',
        'christmas_trees',
        'cleaning_products_supplier',
        'clothing_rental',
        'deck_and_railing_sales_service',
        'do_it_yourself_store',
        'door_sales_service',
        'ethical_grocery',
        'farm_equipment_and_supply',
        'fastener_supplier',
        'firewood',
        'flooring_store',
        'golf_cart_dealer',
        'granite_supplier',
        'hydraulic_equipment_supplier',
        'ice_supplier',
        'lighting_fixtures_and_equipment',
        'lumber_store',
        'mattress_manufacturing',
        'orthopedic_shoe_store',
        'promotional_products_and_services',
        't_shirt_store',
        'taxidermist',
        'tile_store',
        'video_and_video_game_rentals',
        'welding_supply_store',
        'window_supplier',
        'recreation_vehicle_repair',
        'quay',
        'motorcycle_manufacturer',
    ],

    # DINING & DRINKING SERVICE
    'food_beverage': [
        # Restaurants (Cuisine Specific)
        'afghan_restaurant', 'african_restaurant', 'american_restaurant', 'argentine_restaurant',
        'asian_restaurant', 'australian_restaurant', 'belgian_restaurant', 'brazilian_restaurant',
        'british_restaurant', 'cajun_creole_restaurant', 'canadian_restaurant', 'caribbean_restaurant',
        'chinese_restaurant', 'cuban_restaurant', 'ethiopian_restaurant', 'filipino_restaurant',
        'french_restaurant', 'german_restaurant', 'greek_restaurant', 'himalayan_nepalese_restaurant',
        'indian_restaurant', 'irish_restaurant', 'italian_restaurant', 'jamaican_restaurant',
        'japanese_restaurant', 'korean_restaurant', 'kurdish_restaurant', 'lebanese_restaurant',
        'malaysian_restaurant', 'mediterranean_restaurant', 'mexican_restaurant',
        'middle_eastern_restaurant', 'moroccan_restaurant', 'pakistani_restaurant',
        'pan_asian_restaurant', 'persian_iranian_restaurant', 'peruvian_restaurant',
        'polish_restaurant', 'russian_restaurant', 'southern_restaurant', 'spanish_restaurant',
        'sushi_restaurant', 'syrian_restaurant', 'taiwanese_restaurant', 'thai_restaurant',
        'turkish_restaurant', 'venezuelan_restaurant', 'vietnamese_restaurant',

        # Restaurants (Type Specific)
        'bar_and_grill_restaurant', 'barbecue_restaurant', 'bbq_restaurant', 'bistro',
        'breakfast_and_brunch_restaurant', 'breakfast_brunch_restaurant', 'buffet',
        'buffet_restaurant', 'burger_restaurant', 'cafeteria', 'chicken_restaurant',
        'creperie', 'deli', 'dim_sum_restaurant', 'diner', 'fast_food_restaurant',
        'fish_and_chips_restaurant', 'fondue_restaurant', 'food_court', 'food_stand',
        'food_truck', 'gastropub', 'halal_restaurant', 'hot_dog_restaurant', 'kosher_restaurant',
        'noodle_restaurant', 'noodles_restaurant', 'pancake_house', 'pho_restaurant',
        'pizza_restaurant', 'poutinerie_restaurant', 'ramen_restaurant', 'restaurant',
        'sandwich_shop', 'seafood_restaurant', 'steakhouse', 'tapas_bar', 'tapas_restaurant',
        'vegan_restaurant', 'vegetarian_restaurant', 'wings_restaurant',

        # Cafes, Bakeries & Sweets
        'bagel_shop', 'bakery', 'bubble_tea', 'bubble_tea_shop', 'cafe', 'candy_store',
        'chocolate_shop', 'coffee_shop', 'cupcake_shop', 'desserts', 'donut_shop',
        'frozen_yogurt_shop', 'ice_cream_shop', 'internet_cafe', 'juice_bar', 'milk_bar',
        'patisserie_cake_shop', 'smoothie_juice_bar', 'tea_room',

        # Bars & Drinking
        'bar', 'beach_bar', 'beer_bar', 'brewery', 'champagne_bar', 'cocktail_bar',
        'dive_bar', 'hotel_bar', 'lounge', 'pub', 'sake_bar', 'sports_bar',
        'whiskey_bar', 'wine_bar',

        # Catering
        'bartender', 'caterer', 'catering', 'food_delivery_service', 'meal_prep', 'personal_chef',

        # (added from unmapped_categories.txt)
        'arabian_restaurant',
        'asian_fusion_restaurant',
        'austrian_restaurant',
        'beer_garden',
        'beverage_supplier',
        'brasserie',
        'burmese_restaurant',
        'cambodian_restaurant',
        'catalan_restaurant',
        'chocolatier',
        'coffee_roastery',
        'colombian_restaurant',
        'comfort_food_restaurant',
        'czech_restaurant',
        'delicatessen',
        'distillery',
        'doner_kebab',
        'donuts',
        'dumpling_restaurant',
        'eastern_european_restaurant',
        'eat_and_drink',
        'ecuadorian_restaurant',
        'egyptian_restaurant',
        'empanadas',
        'european_restaurant',
        'falafel_restaurant',
        'food',
        'frozen_yoghurt_shop',
        'gelato',
        'gluten_free_restaurant',
        'hawaiian_restaurant',
        'hong_kong_style_cafe',
        'hookah_bar',
        'hungarian_restaurant',
        'ice_cream_and_frozen_yoghurt',
        'indonesian_restaurant',
        'irish_pub',
        'israeli_restaurant',
        'latin_american_restaurant',
        'molecular_gastronomy_restaurant',
        'mongolian_restaurant',
        'piano_bar',
        'pie_shop',
        'portuguese_restaurant',
        'romanian_restaurant',
        'salad_bar',
        'salsa_club',
        'scandinavian_restaurant',
        'scottish_restaurant',
        'shaved_ice_shop',
        'singaporean_restaurant',
        'soup_restaurant',
        'speakeasy',
        'sri_lankan_restaurant',
        'swiss_restaurant',
        'taco_restaurant',
        'texmex_restaurant',
        'theme_restaurant',
        'waffle_restaurant',
        'winery',
        'armenian_restaurant',
        'asian_grocery_store',
        'bagel_restaurant',
        'bangladeshi_restaurant',
        'basque_restaurant',
        'bulgarian_restaurant',
        'chicken_wings_restaurant',
        'diy_foods_restaurant',
        'friterie',
        'health_food_restaurant',
        'iberian_restaurant',
        'indo_chinese_restaurant',
        'japanese_confectionery_shop',
        'jewish_restaurant',
        'live_and_raw_food_restaurant',
        'meat_restaurant',
        'poke_restaurant',
        'seafood_wholesaler',
        'slovakian_restaurant',
        'soul_food',
        'tiki_bar',
        'ukrainian_restaurant',
        'cajun_and_creole_restaurant',
        'dominican_restaurant',
        'georgian_restaurant',
    ],

    # BUSINESS & FINANCIAL
    'corporate_service': [
        # Financial
        'accountant', 'atms', 'audit_firm', 'auto_insurance', 'auto_loan_provider',
        'bank', 'bank_credit_union', 'banks', 'bookkeeper', 'collection_agencies',
        'currency_exchange', 'financial_advising', 'financial_service',
        'health_insurance_office', 'insurance_agency', 'investment_firm', 'life_insurance',
        'money_transfer_services', 'mortgage_broker', 'stock_broker', 'tax_services',

        # Corporate & Professional Services
        'advertising_agency', 'business', 'business_consulting',
        'business_management_services', 'business_to_business', 'call_center',
        'career_counseling', 'corporate_office', 'coworking_space', 'e_commerce_service',
        'employment_agencies', 'executive_search_consultants', 'graphic_designer',
        'hr_consulting', 'information_technology_company', 'internet_marketing_service',
        'management_consultant', 'marketing_agency', 'marketing_consultant', 'media_agency',
        'media_news_website', 'office_service', 'professional_services', 'public_relations',
        'recruitment_agency', 'software_development', 'talent_agency', 'telemarketing',
        'topic_publisher', 'translation_services', 'virtual_office', 'web_designer',

        # Legal
        'bail_bonds', 'divorce_and_family_law', 'employment_law', 'lawyer',
        'legal_firm', 'legal_services', 'notary', 'notary_public',

        # Real Estate (Office)
        'apartment_agent', 'commercial_real_estate', 'escrow_service', 'housing_cooperative',
        'property_management', 'real_estate', 'real_estate_agent', 'real_estate_appraiser',
        'real_estate_service', 'title_company',

        # (added from unmapped_categories.txt)
        'appellate_practice_lawyers',
        'appraisal_services',
        'automation_services',
        'b2b_apparel',
        'b2b_electronic_equipment',
        'b2b_jewelers',
        'b2b_science_and_technology',
        'b2b_textiles',
        'bank_equipment_service',
        'bankruptcy_law',
        'brokers',
        'business_advertising',
        'business_banking_service',
        'business_financing',
        'business_law',
        'business_records_storage_and_management',
        'business_to_business_services',
        'check_cashing_payday_loans',
        'civil_rights_lawyers',
        'commercial_industrial',
        'commercial_printer',
        'computer_coaching',
        'computer_hardware_company',
        'computer_wholesaler',
        'contract_law',
        'copywriting_service',
        'credit_and_debt_counseling',
        'credit_union',
        'criminal_defense_law',
        'debt_relief_services',
        'disability_law',
        'estate_planning_law',
        'general_litigation',
        'holding_companies',
        'human_resource_services',
        'image_consultant',
        'immigration_assistance_services',
        'immigration_law',
        'installment_loans',
        'international_business_and_trade_services',
        'internet_service_provider',
        'investing',
        'ip_and_internet_law',
        'it_support_and_service',
        'law_schools',
        'mass_media',
        'media_news_company',
        'merchandising_service',
        'online_shop',
        'paralegal_services',
        'patent_law',
        'personal_assistant',
        'personal_injury_law',
        'print_media',
        'private_establishments_and_corporates',
        'radio_station',
        'real_estate_investment',
        'social_media_agency',
        'social_media_company',
        'tax_law',
        'telecommunications_company',
        'telephone_services',
        'television_service_providers',
        'television_station',
        'trusts',
        'web_hosting_service',
        'wills_trusts_and_probate',
        'writing_service',
        'b2b_cleaning_and_waste_management',
        'b2b_food_products',
        'b2b_furniture_and_housewares',
        'b2b_machinery_and_tools',
        'b2b_rubber_and_plastics',
        'b2b_sporting_and_recreation_goods',
        'business_brokers',
        'business_equipment_and_supply',
        'business_signage',
        'business_storage_and_transportation',
        'chambers_of_commerce',
        'consultant_and_general_service',
        'court_reporter',
        'customs_broker',
        'data_recovery',
        'domestic_business_and_trade_organizations',
        'duplication_services',
        'engineering_schools',
        'environmental_abatement_services',
        'fidelity_and_surety_bonds',
        'food_and_beverage_consultant',
        'food_consultant',
        'home_and_rental_insurance',
        'home_staging',
        'inventory_control_service',
        'it_consultant',
        'medical_law',
        'mortgage_lender',
        'newspaper_advertising',
        'packaging_contractors_and_service',
        'payroll_services',
        'private_equity_firm',
        'product_design',
        'public_adjuster',
        'publicity_service',
        'real_estate_law',
        'research_institute',
        'secretarial_services',
        'shredding_services',
        'stock_and_bond_brokers',
        'telemarketing_services',
        'temp_agency',
        'typing_services',
        'valet_service',
        'vacation_rental_agents',
        'social_security_law',
    ],

    # INDUSTRIAL & TRADES
    'industrial_service': [
        # Construction & Trades
        'architect', 'architectural_designer', 'bathroom_remodeling', 'building_contractor',
        'carpenter', 'chemical_plant', 'chimney_sweep', 'construction_services', 'contractor',
        'electrician', 'energy_equipment_and_solution', 'engineering_services',
        'fence_and_gate_sales_service', 'fire_and_water_damage_restoration',
        'fire_protection_service', 'fireplace_service', 'flooring_contractor',
        'flooring_contractors', 'forestry_service', 'gardener', 'general_contractor',
        'glass_and_mirror_sales_service', 'gutter_service', 'handyman', 'home_developer',
        'home_security', 'home_service', 'hvac_services', 'interior_design',
        'junk_removal_and_hauling', 'key_and_locksmith', 'kitchen_remodeling',
        'land_surveying', 'landscaping', 'locksmith', 'masonry', 'painter', 'painting',
        'pest_control', 'pest_control_service', 'plastic_fabrication_company', 'plumber',
        'plumbing', 'roofer', 'roofing', 'security_services', 'tree_services',
        'window_installation', 'window_washing',

        # Wholesale & Manufacturing
        'appliance_manufacturer', 'business_manufacturing_and_supply', 'clothing_company',
        'distributor', 'electrical_wholesaler', 'food_beverage_service_distribution',
        'furniture_manufacturers', 'importer_and_exporter', 'industrial_equipment',
        'industrial_equipment_supplier', 'machine_and_tool_rentals', 'metal_supplier',
        'office_equipment', 'print_shop', 'printing_services', 'screen_printing', 'sign_shop',
        'wholesale_grocer', 'wholesaler', 'wine_wholesaler',

        # (added from unmapped_categories.txt)
        'acoustical_consultant',
        'agricultural_cooperatives',
        'agricultural_service',
        'agriculture',
        'altering_and_remodeling_contractor',
        'antenna_service',
        'bathtub_and_sink_repairs',
        'biotechnology_company',
        'book_magazine_distribution',
        'bookbinding',
        'bottled_water_company',
        'builders',
        'ceiling_and_roofing_repair_and_service',
        'ceiling_service',
        'commercial_refrigeration',
        'countertop_installation',
        'damage_restoration',
        'demolition_service',
        'display_home_center',
        'distribution_services',
        'electric_utility_provider',
        'electricity_supplier',
        'energy_company',
        'environmental_and_ecological_services_for_businesses',
        'excavation_service',
        'farming_services',
        'glass_manufacturer',
        'home_automation',
        'home_inspector',
        'home_window_tinting',
        'hotel_supply_service',
        'hvac_supplier',
        'industrial_company',
        'jewelry_and_watches_manufacturer',
        'laboratory_equipment_supplier',
        'livestock_breeder',
        'machine_shop',
        'masonry_concrete',
        'masonry_contractors',
        'metal_fabricator',
        'metal_materials_and_experts',
        'mining',
        'natural_gas_supplier',
        'oil_and_gas',
        'paving_contractor',
        'plastic_manufacturer',
        'public_utility_company',
        'restaurant_equipment_and_supply',
        'restaurant_wholesale',
        'screen_printing_t_shirt_printing',
        'security_systems',
        'shutters',
        'sign_making',
        'skylight_installation',
        'snow_removal_service',
        'solar_installation',
        'stone_and_masonry',
        'structural_engineer',
        'tableware_supplier',
        'utility_service',
        'vending_machine_supplier',
        'warehouses',
        'washer_and_dryer_repair_service',
        'water_supplier',
        'water_treatment_equipment_and_services',
        'windows_installation',
        'wood_and_pulp',
        'agriculture_association',
        'air_duct_cleaning_service',
        'aircraft_manufacturer',
        'auto_customization',
        'auto_electrical_repair',
        'auto_restoration_services',
        'automotive_wheel_polishing_service',
        'avionics_shop',
        'carpet_installation',
        'casting_molding_and_machining',
        'civil_engineers',
        'clock_repair_service',
        'coal_and_coke',
        'construction_management',
        'crops_production',
        'drywall_services',
        'elevator_service',
        'foundation_repair',
        'furniture_repair',
        'furniture_reupholstery',
        'furniture_wholesalers',
        'garage_door_service',
        'geological_services',
        'glass_blowing',
        'grain_elevators',
        'hydraulic_repair_service',
        'insulation_installation',
        'iron_and_steel_industry',
        'irrigation',
        'junkyard',
        'landscape_architect',
        'lawn_service',
        'logging_services',
        'mechanical_engineers',
        'metals',
        'paper_mill',
        'patio_covers',
        'plasterer',
        'pool_and_hot_tub_services',
        'pool_cleaning',
        'powder_coating_service',
        'power_plants_and_power_plant_service',
        'pressure_washing',
        'refinishing_services',
        'sandblasting_service',
        'scrap_metals',
        'septic_services',
        'siding',
        'stucco_services',
        'tiling',
        'tobacco_company',
        'truck_repair',
        'wallpaper_installers',
        'water_delivery',
        'water_heater_installation_repair',
        'water_purification_services',
        'waterproofing',
        'welders',
        'well_drilling',
        'wheel_and_rim_repair',
        'wildlife_control',
        'mobile_home_repair',
    ],

    # CONSUMER SERVICES
    'consumer_service': [
        # Cleaning & Laundry
        'carpet_cleaning', 'cleaning_services', 'dry_cleaning', 'gents_tailor',
        'home_cleaning', 'laundromat', 'laundry_service', 'laundry_services',
        'leather_repair', 'office_cleaning', 'sewing_and_alterations', 'shoe_repair', 'tailor',

        # Storage & Moving
        'luggage_storage', 'movers', 'moving_company', 'organization', 'self_storage',
        'storage', 'storage_facility',

        # Tech & Media
        'appliance_repair_service', 'broadcasting_media_production', 'event_photography',
        'it_service_and_computer_repair', 'movie_television_studio', 'music_production',
        'photo_booth', 'photo_booth_rental', 'photographer',
        'photography_store_and_services',

        # Pets
        'animal_shelter', 'animal_hospital', 'dog_groomer', 'dog_walker', 'kennel',
        'pet_clinic', 'pet_groomer', 'pet_services', 'veterinarian',

        # Travel Agents
        'passport_and_visa_services', 'tour_operator', 'travel', 'travel_agency',
        'travel_agents', 'travel_services',

        # Death Care
        'cemetery', 'cemeteries', 'crematory', 'funeral_home',
        'funeral_services_and_cemeteries',

        # BEAUTY & WELLNESS
        'barber', 'barber_shop', 'beauty_and_spa', 'beauty_salon', 'day_spa',
        'hair_loss_center', 'hair_removal', 'hair_salon', 'hair_stylist', 'kids_hair_salon',
        'laser_hair_removal', 'makeup_artist', 'massage', 'nail_salon', 'permanent_makeup',
        'sauna', 'skin_care', 'spas', 'tanning_salon', 'tattoo_and_piercing',
        'tattoo_removal', 'teeth_whitening', 'waxing',

        # Misc
        'life_coach',

        # (added from unmapped_categories.txt)
        'animation_studio',
        'aromatherapy',
        'astrologer',
        'audiovisual_equipment_rental',
        'dj_service',
        'dog_trainer',
        'dog_walkers',
        'electronics_repair_shop',
        'estate_liquidation',
        'eyelash_service',
        'fortune_telling_service',
        'furniture_assembly',
        'furniture_rental_service',
        'garbage_collection_service',
        'genealogists',
        'hair_extensions',
        'health_spa',
        'janitorial_services',
        'jewelry_repair_service',
        'magician',
        'marriage_or_relationship_counselor',
        'matchmaker',
        'medical_spa',
        'mobile_phone_repair',
        'music_production_services',
        'musical_instrument_services',
        'nanny_services',
        'pet_boarding',
        'pet_breeder',
        'pets',
        'petting_zoo',
        'piercing',
        'private_investigation',
        'psychic',
        'record_label',
        'recording_and_rehearsal_studio',
        'self_storage_facility',
        'session_photography',
        'sightseeing_tour_agency',
        'threading_service',
        'ticket_sales',
        'translating_and_interpreting_services',
        'travel_company',
        'video_film_production',
        'videographer',
        'watch_repair_service',
        'wedding_chapel',
        'wedding_planning',
        'adoption_services',
        'animal_rescue_service',
        'art_restoration',
        'art_restoration_service',
        'blow_dry_blow_out_service',
        'boudoir_photography',
        'calligraphy',
        'car_inspection',
        'car_window_tinting',
        'certification_agency',
        'commissioned_artist',
        'cremation_services',
        'digitizing_services',
        'embroidery_and_crochet',
        'engraving',
        'eyebrow_service',
        'family_counselor',
        'family_service_center',
        'fingerprinting_service',
        'foster_care_services',
        'gold_buyer',
        'goldsmith',
        'hair_replacement',
        'halfway_house',
        'home_organization',
        'house_sitting',
        'hypnosis_hypnotherapy',
        'indoor_landscaping',
        'inspection_services',
        'mailbox_center',
        'media_critic',
        'media_restoration_service',
        'mediator',
        'packing_services',
        'personal_care_service',
        'pet_cemetery_and_crematorium_services',
        'pet_sitting',
        'piano_services',
        'process_servers',
        'shoe_shining_service',
        'snuggle_service',
        'tv_mounting',
        'video_game_critic',
    ],

    # HEALTHCARE
    'healthcare': [
        # Practitioners
        'allergist', 'anesthesiologist', 'cardiologist', 'chiropractor',
        'cosmetic_surgeon', 'dentist', 'dermatologist', 'doctor', 'endocrinologist',
        'endodontist', 'eyewear_and_optician', 'family_practice', 'gastroenterologist',
        'general_dentistry', 'general_practitioner', 'hematologist',
        'infectious_disease_specialist', 'nephrologist', 'neurologist', 'nutritionist',
        'obstetrician_and_gynecologist', 'occupational_therapist', 'oncologist',
        'ophthalmologist', 'optometrist', 'oral_surgeon', 'orthodontist',
        'orthopedic_surgeon', 'pain_management', 'pediatrician', 'physical_therapist',
        'physical_therapy', 'physiotherapist', 'plastic_surgeon', 'podiatrist',
        'psychiatrist', 'psychologist', 'pulmonologist', 'radiologist', 'rheumatologist',
        'sports_medicine', 'urologist',

        # Facilities
        'blood_and_plasma_donation_center', 'blood_bank', 'clinic', 'clinical_laboratories',
        'diagnostic_services', 'emergency_room', 'health_and_medical', 'hospital',
        'laboratory', 'medical_center', 'medical_research_and_development',
        'outpatient_clinic', 'public_health_clinic', 'rehabilitation_center', 'urgent_care',

        # Senior & Care
        'adult_day_care', 'assisted_living', 'assisted_living_facility', 'home_health_care',
        'hospice', 'nursing_home', 'retirement_home', 'skilled_nursing',

        # Mental Health
        'addiction_rehabilitation_center', 'alcohol_and_drug_treatment_centers',
        'counseling', 'counseling_and_mental_health', 'counselor',
        'mental_health_services', 'psychotherapy', 'therapist',

        # Alternative
        'acupuncture', 'alternative_medicine', 'herbalist', 'homeopath', 'massage_therapy',
        'naturopath', 'naturopathic_holistic', 'osteopath', 'reflexology', 'reiki',

        # Pharmacy & Supplies
        'drugstore', 'medical_supply', 'medical_supply_store', 'mobility_equipment_services',
        'orthotics', 'pharmacy', 'prosthetics', 'weight_loss_center',

        # (added from unmapped_categories.txt)
        'abuse_and_addiction_treatment',
        'audiologist',
        'cannabis_clinic',
        'cannabis_dispensary',
        'childrens_hospital',
        'cosmetic_dentist',
        'ear_nose_and_throat',
        'environmental_testing',
        'eye_care_clinic',
        'fertility',
        'gerontologist',
        'health_department',
        'internal_medicine',
        'laboratory_testing',
        'maternity_centers',
        'medical_school',
        'medical_service_organizations',
        'meditation_center',
        'nurse_practitioner',
        'occupational_medicine',
        'occupational_safety',
        'orthopedist',
        'osteopathic_physician',
        'pharmaceutical_companies',
        'pharmaceutical_products_wholesaler',
        'psychotherapist',
        'sex_therapist',
        'speech_therapist',
        'surgeon',
        'surgical_appliances_and_supplies',
        'walk_in_clinic',
        'womens_health_clinic',
        'dialysis_clinic',
        'emergency_roadside_service',
        'geriatric_medicine',
        'health_coach',
        'health_consultant',
        'health_retreats',
        'hematology',
        'medical_transportation',
        'nursing_school',
        'occupational_therapy',
        'pathologist',
        'pediatric_dentist',
        'prenatal_perinatal_care',
        'surgical_center',
        'urgent_care_clinic',
        'dental_laboratories',
        'dietitian',
        'wellness_program',
    ],

    # EDUCATION
    'education': [
        # Schools
        'art_school', 'boarding_school', 'business_schools', 'college', 'college_university',
        'community_college', 'cooking_school', 'cosmetology_school', 'drama_school',
        'driving_school', 'educational_services', 'elementary_school', 'flight_school',
        'high_school', 'kindergarten', 'language_school', 'law_school', 'massage_school',
        'medical_sciences_schools', 'middle_school', 'montessori_school', 'music_school',
        'nursery', 'preschool', 'private_school', 'public_school', 'religious_school',
        'school', 'science_schools', 'special_education_school', 'technical_school',
        'trade_school', 'university', 'vocational_school', 'waldorf_school', 'academic_library',

        # Training & Support 'art_classes', 'barber_school', 'campus_building',
        'child_care_and_day_care', 'childcare', 'cpr_classes', 'dance_school',
        'day_care_preschool', 'education', 'first_aid_training', 'food_safety_training',
        'learning_center', 'music_lessons', 'private_tutor', 'study_hall', 'test_prep',
        'training_center', 'tutoring_center', 'campus_building', 'student_union',

        # (added from unmapped_categories.txt)
        'adult_education',
        'after_school_program',
        'bartending_school',
        'computer_museum',
        'educational_camp',
        'educational_research_institute',
        'indoor_playcenter',
        'ski_and_snowboard_school',
        'specialty_school',
        'test_preparation',
        'university_housing',
        'vocational_and_technical_school',
        'circus_school',
        'first_aid_class',
        'scuba_diving_instruction',
        'sports_and_fitness_instruction',
        'traffic_school',
    ],

    # GOVERNMENT & COMMUNITY
    'public_service': [
        # Government
        'central_government_office', 'city_hall', 'consulate', 'courthouse', 'embassy',
        'fire_department', 'fire_station', 'government_building', 'government_services',
        'jail', 'jail_and_prison', 'law_enforcement', 'military_base', 'police_department',
        'police_station', 'post_office', 'prison', 'public_service_and_government',

        # Community
        'charity_organization', 'community_center',
        'community_services_non_profits', 'food_bank', 'food_banks', 'homeless_shelter',
        'library', 'public_bathroom', 'public_library', 'scout_hall', 'senior_center',
        'social_and_human_services', 'social_service_organizations', 'youth_organizations',

        # (added from unmapped_categories.txt)
        'ambulance_and_ems_services',
        'armed_forces_branch',
        'automobile_registration_service',
        'disability_services_and_support_organization',
        'environmental_conservation_and_ecological_organizations',
        'environmental_conservation_organization',
        'housing_authorities',
        'labor_union',
        'local_and_state_government_offices',
        'non_governmental_association',
        'political_organization',
        'political_party_office',
        'private_association',
        'public_and_government_association',
        'public_bath_houses',
        'public_restrooms',
        'public_toilet',
        'recycling_center',
        'senior_citizen_services',
        'town_hall',
        'civic_center',
        'community_gardens',
        'department_of_motor_vehicles',
        'dui_law',
        'federal_government_offices',
        'registry_office',
        'social_security_services',
        'volunteer_association',
        'child_protection_service',
        'unemployment_office',
    ],

    # TRANSPORT & AUTO
    'transportation_facility': [
        # Terminals & Stops
        'airport', 'airport_lounge', 'airport_terminal', 'airstrip', 'bus_station',
        'bus_stop', 'commuter_rail_station', 'ferry_terminal', 'heliport', 'metro_station',
        'pier', 'port', 'public_transportation', 'subway_station', 'train_station',
        'tram_stop', 'transportation',

        # Rental & Fleet
        'ambulance_service', 'bike_rental', 'boat_rental_and_training', 'bus_charter',
        'car_rental', 'car_rental_agency', 'limo_services', 'limousine_service',
        'private_car_service', 'rental_service', 'rental_services', 'rideshare',
        'scooter_rental', 'shuttle_service', 'taxi_service', 'taxi_stand', 'truck_rental',
        'van_rental', 'yacht_charter',

        # Logistics (Physical)
        'courier', 'courier_and_delivery_services', 'freight_and_cargo_service',
        'freight_forwarding_agency', 'motor_freight_trucking', 'package_locker',
        'shipping_center', 'trucking_company',

        # (added from unmapped_categories.txt)
        'airline',
        'airlines',
        'bike_rentals',
        'bus_rentals',
        'bus_service',
        'canal',
        'ferry_boat_company',
        'ferry_service',
        'heliports',
        'light_rail_and_subway_stations',
        'marina',
        'railroad_freight',
        'railway_service',
        'ride_sharing',
        'truck_rentals',
        'airport_shuttles',
        'balloon_ports',
        'balloon_services',
        'boat_charter',
        'boat_tours',
        'canoe_and_kayak_hire_service',
        'car_broker',
        'car_buyer',
        'car_sharing',
        'coach_bus',
        'fishing_charter',
        'motorcycle_rentals',
        'scooter_dealers',
        'shipping_collection_services',
        'trailer_rentals',
        'truck_dealer_for_businesses',
        'rv_rentals',
    ],

    'automotive_facility': [
        # Service
        'auto_body_shop', 'auto_detailing', 'auto_glass_services', 'auto_repair',
        'auto_upholstery', 'automotive', 'automotive_repair',
        'automotive_services_and_repair', 'brake_service_and_repair', 'car_wash',
        'emissions_inspection', 'mechanic', 'motorcycle_repair', 'oil_change_station',
        'tire_dealer_and_repair', 'tire_repair_shop', 'tire_shop', 'towing_service',
        'transmission_repair', 'vehicle_inspection',

        # Sales
        'auto_manufacturers_and_distributors', 'automotive_parts_and_accessories',
        'car_dealer', 'motorcycle_dealer', 'part_store', 'recreational_vehicle_dealer',
        'trailer_dealer', 'truck_dealer', 'used_car_dealer',

        # Infrastructure
        'charging_station', 'ev_charging_station', 'fuel_station', 'gas_station',
        'parking', 'parking_garage', 'parking_lot', 'petrol_station', 'valet_parking',

        # (added from unmapped_categories.txt)
        'auto_company',
        'auto_glass_service',
        'automotive_consultant',
        'automotive_dealer',
        'bike_repair_maintenance',
        'boat_dealer',
        'boat_service_and_repair',
        'car_stereo_store',
        'engine_repair_service',
        'mobile_home_dealer',
        'aircraft_dealer',
        'automobile_leasing',
        'automotive_storage_facility',
        'commercial_vehicle_dealer',
        'dumpster_rentals',
        'motorsport_vehicle_dealer',
    ],

    # ACCOMMODATION
    'hotel_lodging': [
        'accommodation', 'bed_and_breakfast', 'cabin', 'campground', 'cottage',
        'guest_house', 'hostel', 'hotel', 'inn', 'lodging', 'motel', 'resort', 'rv_park',
        'vacation_rental',

        # (added from unmapped_categories.txt)
        'apartments',
        'condominium',
        'holiday_rental_home',
        'lodge',
        'pension',
        'service_apartments',
        'shared_office_space',
        'beach_resort',
        'mobile_home_park',
        'self_catering_accommodation',
    ],

    # RECREATION & NATURE
    'sports_fitness': [
        'active_life', 'arena', 'athletic_field', 'baseball_field', 'basketball_court',
        'bowling_alley', 'boxing_gym', 'climbing_gym', 'cricket_ground', 'driving_range',
        'fitness_center', 'fitness_trainer', 'football_club', 'football_stadium',
        'golf_club', 'golf_course', 'gym', 'ice_skating', 'leisure_center',
        'martial_arts_club', 'martial_arts_school', 'miniature_golf_course', 'pilates_studio',
        'public_swimming_pool', 'racetrack', 'recreation_center', 'rock_climbing',
        'roller_skating', 'rugby_club', 'skate_park', 'skating_rink', 'soccer_club',
        'soccer_stadium', 'sports_and_recreation_venue', 'sports_club_and_league',
        'sports_field', 'stadium', 'stadium_arena', 'swimming_instructor', 'swimming_pool',
        'tennis_court', 'yoga_studio',

        # (added from unmapped_categories.txt)
        'amateur_sports_league',
        'amateur_sports_team',
        'baseball_stadium',
        'boot_camp',
        'cycling_classes',
        'disc_golf_course',
        'flyboarding_center',
        'golf_instructor',
        'gymnastics_center',
        'kiteboarding',
        'pool_billiards',
        'pool_hall',
        'professional_sports_league',
        'professional_sports_team',
        'race_track',
        'rock_climbing_gym',
        'school_sports_league',
        'school_sports_team',
        'scuba_diving_center',
        'shooting_range',
        'ski_resort',
        'sky_diving',
        'soccer_field',
        'table_tennis_club',
        'track_stadium',
        'adventure_sports_center',
        'archery_range',
        'badminton_court',
        'barre_classes',
        'basketball_stadium',
        'batting_cage',
        'boxing_class',
        'country_club',
        'equestrian_facility',
        'go_kart_club',
        'hockey_arena',
        'hockey_field',
        'horse_riding',
        'horse_trainer',
        'horseback_riding_service',
        'ice_skating_rink',
        'kickboxing_club',
        'mountain_bike_trails',
        'officiating_services',
        'paddleboarding_center',
        'racquetball_court',
        'rafting_kayaking_area',
        'rock_climbing_spot',
        'roller_skating_rink',
        'rugby_pitch',
        'rugby_stadium',
        'sailing_area',
        'sailing_club',
        'squash_court',
        'surfboard_rental',
        'surfing',
        'taekwondo_club',
        'tai_chi_studio',
        'tennis_stadium',
    ],

    'park': [
        'beach', 'botanical_garden', 'city_park', 'dog_park', 'forest', 'garden',
        'hiking_trail', 'lake', 'national_park', 'nature_reserve', 'park', 'playground',
        'state_park', 'wildlife_sanctuary',

        # (added from unmapped_categories.txt)
        'backpacking_area',
        'farm',
        'fountain',
        'island',
        'memorial_park',
        'mountain',
        'plaza',
        'public_plaza',
        'river',
        'urban_farm',
        'cave',
        'fort',
        'lighthouse',
        'natural_hot_springs',
        'observatory',
        'waterfall',
    ],

    # ARTS, ENTERTAINMENT & RELIGION
    'arts_culture': [
        'aquarium', 'art_gallery', 'art_museum', 'attractions_and_activities', 'castle',
        'childrens_museum', 'cultural_center', 'historic_site', 'historical_place',
        'history_museum', 'landmark_and_historical_building', 'memorial', 'monument',
        'museum', 'planetarium', 'science_center', 'science_museum', 'sculpture_statue',
        'tourist_information', 'tours', 'visitor_center', 'zoo',

        # (added from unmapped_categories.txt)
        'architectural_tours',
        'architecture',
        'atv_rentals_and_tours',
        'bridge',
        'convents_and_monasteries',
        'fair',
        'modern_art_museum',
        'palace',
        'street_art',
        'structure_and_geography',
        'tower',
        'community_museum',
        'craft_shop',
        'esports_team',
        'feng_shui',
        'general_festivals',
        'haunted_house',
        'onsen',
        'textile_museum',
        'film_festivals_and_organizations',
    ],

    'entertainment': [
        # Venues & Events
        'amphitheater', 'arts_and_entertainment', 'banquet_hall', 'cinema', 'club',
        'concert_hall', 'corporate_entertainment_services', 'drive_in_theater',
        'event_planning', 'event_technology_service', 'event_venue', 'fraternal_organization',
        'movie_theater', 'music_venue', 'opera_house', 'party_and_event_planning',
        'party_equipment_rental', 'performing_arts_venue', 'social_club', 'theater',
        'theatre', 'topic_concert_venue', 'venue_and_event_space', 'wedding_venue',

        # Nightlife
        'adult_entertainment', 'comedy_club', 'dance_club', 'gay_bar', 'jazz_and_blues',
        'karaoke', 'nightclub', 'nightlife',

        # Gaming & Fun
        'amusement_park', 'arcade', 'betting_center', 'billiards_or_pool_hall', 'bingo_hall',
        'bookmakers', 'carnival', 'casino', 'circus', 'escape_room', 'game_room',
        'go_kart_track', 'kids_recreation_and_party', 'laser_tag', 'paintball',
        'theme_park', 'trampoline_park', 'video_arcade', 'water_park',

        # (added from unmapped_categories.txt)
        'auction_house',
        'auditorium',
        'choir',
        'country_dance_hall',
        'escape_rooms',
        'game_publisher',
        'horse_boarding',
        'hot_tubs_and_pools',
        'musician',
        'opera_and_ballet',
        'performing_arts',
        'strip_club',
        'theaters_and_performance_venues',
        'theatrical_productions',
        'bail_bonds_service',
        'climbing_service',
        'eatertainment',
        'fishing_club',
        'night_market',
        'psychic_medium',
        'trivia_host',
    ],

    'religion': [
        'anglican_church', 'buddhist_temple', 'cathedral', 'catholic_church', 'chapel',
        'church', 'church_cathedral', 'convent', 'hindu_temple',
        'jehovahs_witness_kingdom_hall', 'kingdom_hall', 'monastery', 'mosque',
        'religious_organization', 'shrine', 'sikh_temple', 'synagogue', 'temple',

        # (added from unmapped_categories.txt)
        'mission',
        'pentecostal_church',
        'religious_destination',
        'baptist_church',
        'evangelical_church',
        'shinto_shrines',
    ]
}

# HELPER: REVERSE LOOKUP
def get_reverse_mapping(mapping_dict):
    """
    Generates a dictionary where Keys = Specific Types (e.g. 'sushi')
    and Values = Category (e.g. 'food_service').
    Checks for duplicates.
    """
    reverse_map = {}
    duplicates = []

    for category, items in mapping_dict.items():
        for item in items:
            if item in reverse_map:
                duplicates.append(f"{item} (found in {reverse_map[item]} and {category})")
            reverse_map[item] = category

    if duplicates:
        print(f"Warning: Found duplicates in mapping: {duplicates}")

    return reverse_map

CATEGORY_TO_CLASS = get_reverse_mapping(CATEGORY_MAPPING)
_unmapped_categories: set[str] = set()   # collect categories not in CATEGORY_MAPPING

def classify_poi(cats):
    """Classify a POI based on its categories dict."""
    if pd.isna(cats) or not isinstance(cats, dict):
        return 'other'
    
    # Check primary category first
    primary = cats.get('primary', '')
    if primary in CATEGORY_TO_CLASS:
        return CATEGORY_TO_CLASS[primary]
    
    # Check alternate categories
    alt = cats.get('alternate', None)
    if alt is not None:
        if isinstance(alt, np.ndarray):
            alt = alt.tolist()
        elif isinstance(alt, list):
            pass
        else:
            # Fallback for unexpected types
            if primary:
                _unmapped_categories.add(primary)
            return 'other'
            
        for a in alt:
            if a in CATEGORY_TO_CLASS:
                return CATEGORY_TO_CLASS[a]
    
    # None of the categories matched — record them all as unmapped
    if primary:
        _unmapped_categories.add(primary)
    if alt:
        for a in alt:
            _unmapped_categories.add(a)
    return 'other'
def save_unmapped_categories():
    global _unmapped_categories
    if _unmapped_categories:
        unmapped_path = os.path.join(DATA_DIR, "unmapped_categories.txt")
        # Load existing entries so we never lose previously recorded categories
        existing: set[str] = set()
        if os.path.exists(unmapped_path):
            with open(unmapped_path, "r") as f:
                existing = {line.strip() for line in f if line.strip()}
        new_cats = _unmapped_categories - existing
        if new_cats:
            combined = sorted(existing | _unmapped_categories)
            with open(unmapped_path, "w") as f:
                for cat in combined:
                    f.write(cat + "\n")
            print(f"⚠  {len(new_cats)} new unmapped categories added to {unmapped_path} "
                  f"(total: {len(combined)})")
        else:
            print(f"ℹ  {len(_unmapped_categories)} unmapped categories already recorded in {unmapped_path}")

# ------------------------------------------------------------------ #
# Land-use classification
# ------------------------------------------------------------------ #

# Define Category Mapping (Class -> Functional Group)
LAND_USE_MAPPING = {
    'residential': ['residential', 'garages'],
    'commercial': ['commercial', 'retail', 'plaza', 'entertainment'],
    'industrial': ['industrial', 'works', 'brownfield', 'construction', 'landfill', 'resource_extraction'],
    'public_services': ['school', 'university', 'college', 'hospital', 'clinic', 'doctors', 'religious', 'military', 'base', 'cemetery', 'grave_yard', 'education', 'medical'],
    'green_space': ['park', 'grass', 'garden', 'meadow', 'nature_reserve', 'village_green', 'green', 'greenfield', 'flowerbed', 'orchard', 'allotments', 'recreation_ground', 'playground', 'pitch', 'stadium', 'golf_course', 'fairway', 'tee', 'bunker', 'rough', 'water_hazard', 'campground', 'golf', 'managed', 'protected', 'recreation'],
    'transportation': ['railway', 'track', 'pedestrian', 'transportation'],
    'agricultural': ['farmland', 'farmyard', 'agriculture', 'horticulture'],
    'developed': ['developed']
}

LAND_USE_TO_CLASS = get_reverse_mapping(LAND_USE_MAPPING)
_unmapped_land_use: set[str] = set()

def classify_land_use(land_use_class):
    if pd.isna(land_use_class):
        return 'other'
    land_use_class = str(land_use_class)
    if land_use_class in LAND_USE_TO_CLASS:
        return LAND_USE_TO_CLASS[land_use_class]
    _unmapped_land_use.add(land_use_class)
    return 'other'

def compute_land_use_ratios(
    hex_gdf,
    landuse_gdf,
    hex_geometry_col="hex_polygon",
    landuse_category_col="landuse_category",
    ratio_prefix="land_use",
):
    polygon_geometry_types = {"Polygon", "MultiPolygon"}

    hex_polygons = hex_gdf.set_geometry(hex_geometry_col)[[hex_geometry_col]].copy().reset_index()
    hex_polygons = hex_polygons.rename(columns={hex_geometry_col: "geometry"})
    hex_polygons = gpd.GeoDataFrame(hex_polygons, geometry="geometry", crs=hex_gdf.crs)
    hex_polygons["geometry"] = hex_polygons.geometry.make_valid()
    hex_polygons = hex_polygons.explode(index_parts=False).reset_index(drop=True)
    hex_polygons = hex_polygons[
        hex_polygons.geometry.geom_type.isin(polygon_geometry_types)
    ].copy()
    hex_polygons["hex_area"] = hex_polygons.geometry.area
    hex_areas = hex_polygons.set_index("h3_index")["hex_area"]

    empty_ratios = pd.DataFrame(index=hex_gdf.index)
    default_landuse = pd.Series("other", index=hex_gdf.index, name="landuse")

    if landuse_gdf.empty:
        return empty_ratios, default_landuse

    landuse_polygons = landuse_gdf[[landuse_category_col, "geometry"]].copy()
    landuse_polygons = landuse_polygons[landuse_polygons.geometry.notna()].copy()
    landuse_polygons["geometry"] = landuse_polygons.geometry.make_valid()
    landuse_polygons = landuse_polygons.explode(index_parts=False).reset_index(drop=True)
    landuse_polygons = landuse_polygons[
        landuse_polygons.geometry.geom_type.isin(polygon_geometry_types)
    ].copy()
    if landuse_polygons.empty:
        return empty_ratios, default_landuse

    intersections = gpd.overlay(
        hex_polygons[["h3_index", "geometry", "hex_area"]],
        landuse_polygons,
        how="intersection",
        keep_geom_type=False,
    )
    intersections = intersections[
        intersections.geometry.notna() & ~intersections.geometry.is_empty
    ].copy()
    if intersections.empty:
        return empty_ratios, default_landuse

    intersections["landuse"] = intersections[landuse_category_col].apply(classify_land_use)
    intersections["overlap_area"] = intersections.geometry.area

    landuse_ratios = (
        intersections.groupby(["h3_index", "landuse"])["overlap_area"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(hex_gdf.index, fill_value=0.0)
        .div(hex_areas, axis=0)
        .fillna(0.0)
    )

    row_sums = landuse_ratios.sum(axis=1)
    overfilled = row_sums > 1.0
    if overfilled.any():
        landuse_ratios.loc[overfilled] = landuse_ratios.loc[overfilled].div(row_sums[overfilled], axis=0)

    dominant_landuse = landuse_ratios.idxmax(axis=1).where(landuse_ratios.sum(axis=1) > 0, "other")
    dominant_landuse.name = "landuse"

    landuse_ratios.columns = [f"{ratio_prefix}_{column}_ratio" for column in landuse_ratios.columns]
    return landuse_ratios, dominant_landuse.reindex(hex_gdf.index).fillna("other")

def save_unmapped_land_use_categories():
    global _unmapped_land_use
    if _unmapped_land_use:
        unmapped_path = os.path.join(DATA_DIR, "unmapped_land_use_categories.txt")
        existing: set[str] = set()
        if os.path.exists(unmapped_path):
            with open(unmapped_path, "r") as f:
                existing = {line.strip() for line in f if line.strip()}
        new_cats = _unmapped_land_use - existing
        if new_cats:
            combined = sorted(existing | _unmapped_land_use)
            with open(unmapped_path, "w") as f:
                for cat in combined:
                    f.write(cat + "\n")
            print(f"Added {len(new_cats)} new land use categories to {unmapped_path} (total: {len(combined)})")
        else:
            print(f"{len(_unmapped_land_use)} land use categories already recorded in {unmapped_path}")