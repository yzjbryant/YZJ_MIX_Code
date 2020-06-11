def look_for_key(main_box):
    pile = main_box.make_a_pile_to_look_through()
    while pile is not empty:
        box = pile.grab_a_box()
        for item in box:
            if item.is_a_box():
                pile.append(item)
            elif item.is_a_key():
                print "found the key"

def look_for_key(box):
    for item in box:
        if item.is_a_box():
            look_for_key(item) #递归
        elif item.is_a_key():
            print "found the key"



