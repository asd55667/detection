class Config:

    META_DIR = '/media/wcw/TOSHIBA_X/OID/'
    TRAIN_DIR  = META_DIR + 'train/'
    VALIDATION_DIR  = META_DIR + 'validation/'
    TEST_DIR = META_DIR + 'test/'
    OUTPUT_DIR = '/media/wcw/TOSHIBA_X/OID/output/'
    LEVEL_1_DIR = '/media/wcw/TOSHIBA_X/OID/output/level_1_files/'

    MIN_SIZE = 768 
    MAX_SIZE = 1024

    BATCH_SIZE = 1

    MULTI_SCALE_TESTING = False


    RPN_BATCH_SIZE = 1
    FG_RATIO = .5
    POS_THRESH = .5
    NEG_THRESH = .5
    RPN_MIN_SIZE = 1 
    PRE_NMS_N = 1
    POST_NMS_N = 1
    NMS_THRESH = .5
    ANCHOR_RATIO = [0.5, 1, 2]
    ANCHOR_SCALE = [8, 16, 32]
    FEAT_STRIDE = 16
    FG_THRESH = .5
    BG_THREH_LOW = .3
    BG_THREH_HIGH = .5





    LEVEL_1_LABEL = ['Accordion', 'Adhesive tape', 'Airplane', 'Alarm clock', 'Alpaca', 'Ambulance', 'Ant', 'Antelope',
                    'Apple', 'Artichoke', 'Asparagus', 'Backpack', 'Bagel', 'Balloon', 'Banana', 'Barge', 'Barrel',
                    'Baseball bat', 'Baseball glove', 'Bat', 'Bathroom cabinet', 'Bathtub', 'Beaker', 'Bee', 'Beehive',
                    'Beer', 'Bell pepper', 'Belt', 'Bench', 'Bicycle', 'Bicycle helmet', 'Bicycle wheel', 'Bidet',
                    'Billboard', 'Billiard table', 'Binoculars', 'Blender', 'Blue jay', 'Book', 'Bookcase', 'Boot',
                    'Bottle', 'Bow and arrow', 'Bowl', 'Box', 'Boy', 'Brassiere', 'Bread', 'Briefcase', 'Broccoli',
                    'Bronze sculpture', 'Brown bear', 'Bull', 'Burrito', 'Bus', 'Bust', 'Butterfly', 'Cabbage',
                    'Cabinetry', 'Cake', 'Cake stand', 'Camel', 'Camera', 'Canary', 'Candle', 'Candy', 'Cannon',
                    'Canoe', 'Carrot', 'Cart', 'Castle', 'Cat', 'Caterpillar', 'Cattle', 'Ceiling fan', 'Cello',
                    'Centipede', 'Chair', 'Cheetah', 'Chest of drawers', 'Chicken', 'Chopsticks', 'Christmas tree',
                    'Coat', 'Cocktail', 'Coconut', 'Coffee', 'Coffee cup', 'Coffee table', 'Coffeemaker', 'Coin',
                    'Common fig', 'Computer keyboard', 'Computer monitor', 'Computer mouse', 'Convenience store',
                    'Cookie', 'Corded phone', 'Countertop', 'Cowboy hat', 'Crab', 'Cricket ball', 'Crocodile',
                    'Croissant', 'Crown', 'Crutch', 'Cucumber', 'Cupboard', 'Curtain', 'Cutting board', 'Dagger',
                    'Deer', 'Desk', 'Dice', 'Digital clock', 'Dinosaur', 'Dog', 'Dog bed', 'Doll', 'Dolphin',
                    'Door', 'Door handle', 'Doughnut', 'Dragonfly', 'Drawer', 'Dress', 'Drinking straw', 'Drum',
                    'Duck', 'Dumbbell', 'Eagle', 'Earrings', 'Egg', 'Elephant', 'Envelope', 'Falcon', 'Fedora',
                    'Filing cabinet', 'Fire hydrant', 'Fireplace', 'Flag', 'Flashlight', 'Flowerpot', 'Flute',
                    'Food processor', 'Football', 'Football helmet', 'Fork', 'Fountain', 'Fox', 'French fries',
                    'Frog', 'Frying pan', 'Gas stove', 'Giraffe', 'Girl', 'Glasses', 'Goat', 'Goggles', 'Goldfish',
                    'Golf ball', 'Golf cart', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guacamole', 'Guitar',
                    'Hamburger', 'Hamster', 'Handbag', 'Handgun', 'Harbor seal', 'Harp', 'Harpsichord', 'Headphones',
                    'Helicopter', 'High heels', 'Honeycomb', 'Horn', 'Horse', 'Hot dog', 'House', 'Houseplant',
                    'Human arm', 'Human beard', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
                    'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream', 'Infant bed',
                    'Jacket', 'Jaguar', 'Jeans', 'Jellyfish', 'Jet ski', 'Jug', 'Juice', 'Kangaroo', 'Kettle',
                    'Kitchen & dining room table', 'Kitchen knife', 'Kite', 'Knife', 'Ladder', 'Ladybug', 'Lamp',
                    'Lantern', 'Laptop', 'Lavender', 'Lemon', 'Leopard',  'Light bulb', 'Light switch', 'Lifejacket',
                    'Lighthouse', 'Lily', 'Limousine', 'Lion', 'Lizard', 'Lobster', 'Loveseat', 'Lynx', 'Man',
                    'Mango', 'Maple', 'Measuring cup', 'Mechanical fan', 'Microphone', 'Microwave oven', 'Miniskirt',
                    'Mirror', 'Missile', 'Mixer', 'Mobile phone', 'Monkey', 'Motorcycle', 'Mouse', 'Muffin', 'Mug',
                    'Mule', 'Mushroom', 'Musical keyboard', 'Nail', 'Necklace', 'Nightstand', 'Oboe', 'Office building',
                    'Orange', 'Organ', 'Ostrich', 'Otter', 'Oven', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Pancake',
                    'Paper towel', 'Parachute', 'Parrot', 'Pasta', 'Peach', 'Pear', 'Pen', 'Penguin', 'Piano',
                    'Picnic basket', 'Picture frame', 'Pig', 'Pillow', 'Pineapple', 'Pitcher', 'Pizza', 'Plastic bag',
                    'Plate', 'Platter', 'Polar bear', 'Pomegranate', 'Popcorn', 'Porch', 'Porcupine', 'Poster',
                    'Potato', 'Power plugs and sockets', 'Pressure cooker', 'Pretzel', 'Printer', 'Pumpkin',
                    'Punching bag', 'Rabbit', 'Raccoon', 'Radish', 'Raven', 'Refrigerator', 'Rhinoceros', 'Rifle',
                    'Ring binder', 'Rocket', 'Roller skates', 'Rose', 'Rugby ball', 'Ruler', 'Salad',
                    'Salt and pepper shakers', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Scissors', 'Scoreboard',
                    'Screwdriver', 'Sea lion', 'Sea turtle', 'Seahorse', 'Seat belt', 'Segway', 'Serving tray',
                    'Sewing machine', 'Shark', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shotgun', 'Shower', 'Shrimp',
                    'Sink', 'Skateboard', 'Ski', 'Skull', 'Skyscraper', 'Slow cooker', 'Snail', 'Snake', 'Snowboard',
                    'Snowman', 'Snowmobile', 'Snowplow', 'Sock', 'Sofa bed', 'Sombrero', 'Sparrow', 'Spatula',
                    'Spider', 'Spoon', 'Sports uniform', 'Squirrel', 'Stairs', 'Starfish', 'Stationary bicycle',
                    'Stool', 'Stop sign', 'Strawberry', 'Street light', 'Stretcher', 'Studio couch',
                    'Submarine sandwich', 'Suit', 'Suitcase', 'Sun hat', 'Sunflower', 'Sunglasses', 'Surfboard',
                    'Sushi', 'Swan', 'Swim cap', 'Swimming pool', 'Swimwear', 'Sword', 'Table tennis racket',
                    'Tablet computer', 'Taco', 'Tank', 'Tap', 'Tart', 'Taxi', 'Tea', 'Teapot', 'Teddy bear',
                    'Television', 'Tennis ball', 'Tennis racket', 'Tent', 'Tiara', 'Tick', 'Tie', 'Tiger', 'Tin can',
                    'Tire', 'Toaster', 'Toilet', 'Toilet paper', 'Tomato', 'Torch', 'Tortoise', 'Towel', 'Tower',
                    'Traffic light', 'Train', 'Training bench', 'Treadmill', 'Tripod', 'Trombone', 'Truck',
                    'Trumpet', 'Turkey', 'Umbrella', 'Van', 'Vase', 'Vehicle registration plate', 'Violin',
                    'Volleyball', 'Waffle', 'Wall clock', 'Washing machine', 'Waste container', 'Watch',
                    'Watermelon', 'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Willow', 'Window',
                    'Window blind', 'Wine', 'Wine glass', 'Winter melon', 'Wok', 'Woman', 'Wood-burning stove',
                    'Woodpecker', 'Wrench', 'Zebra', 'Zucchini']

cfg = Config()            