from policy import Policy


class Policy2352237(Policy):
    def __init__(self):
        self.height_profiles = []
        self.current_plate = 0
        
        self.plate_inventory = []
        self.total_piece_area = 0
        self.total_plate_area = 0
        pass

    def get_action(self, observation, info):
        if info['filled_ratio'] == 0.0:
            self.initialize_height_profiles(observation["stocks"])

        piece_list = observation["products"]
        piece_list = sorted(piece_list, key=lambda x: -(x["size"][0] * x["size"][1]))

        current_dimensions = [0, 0]
        selected_plate = -1
        placement_x, placement_y = 0, 0

        for piece in piece_list:
            if piece["quantity"] == 0: continue

            current_dimensions = piece["size"]
            if (current_dimensions[0] < current_dimensions[1]): 
                current_dimensions[0], current_dimensions[1] = current_dimensions[1], current_dimensions[0]

            min_waste = 1e18
            selected_coords = (-1, -1)
            plate_selection = -1
            rotation_needed = False

            for idx, (plate, plate_num) in enumerate(self.plate_inventory):
                if idx >= self.current_plate: break

                waste_result, (temp_x, temp_y), temp_rotation = self.find_optimal_placement(piece, plate, plate_num)
                if (waste_result < min_waste):
                    min_waste = waste_result
                    placement_x, placement_y = temp_x, temp_y
                    plate_selection = plate_num
                    rotation_needed = temp_rotation
                    break

            while min_waste == 1e18:
                self.current_plate += 1
                width, height = self._get_stock_size_(self.plate_inventory[self.current_plate - 1][0])
                self.total_plate_area += width * height

                waste_result, (temp_x, temp_y), temp_rotation = self.find_optimal_placement(
                    piece, 
                    self.plate_inventory[self.current_plate - 1][0], 
                    self.plate_inventory[self.current_plate - 1][1]
                )
                if (waste_result < 1e18):
                    min_waste = waste_result
                    placement_x, placement_y = temp_x, temp_y
                    plate_selection = self.plate_inventory[self.current_plate - 1][1]
                    rotation_needed = temp_rotation
                    break
            
            if (rotation_needed): 
                piece["size"][0], piece["size"][1] = piece["size"][1], piece["size"][0]
            
            self.update_height_profile(piece, (placement_x, placement_y), 
                                    observation["stocks"][plate_selection], plate_selection)
            
            placement = {
                "stock_idx": plate_selection, 
                "size": current_dimensions, 
                "position": (placement_x, placement_y)
            }

            self.total_piece_area += piece["size"][0] * piece["size"][1]
            return placement 

    class HeightSegment():
        def __init__(self, start_x, end_x, level) -> None:
            self.start_x = start_x
            self.end_x = end_x
            self.level = level

        def has_overlap(self, piece_dims, piece_pos) -> bool:
            piece_start_x, piece_base_y = piece_pos
            piece_width, piece_height = piece_dims
            piece_end_x = piece_start_x + piece_width

            if (piece_end_x <= self.start_x or piece_start_x >= self.end_x): 
                return False

            if (piece_base_y >= self.level): 
                return False
            
            return True
        
        def compute_gap_area(self, piece_dims, piece_pos) -> int:
            piece_start_x, piece_base_y = piece_pos
            piece_width, piece_height = piece_dims
            piece_end_x = piece_start_x + piece_width

            if (piece_end_x <= self.start_x or piece_start_x >= self.end_x): 
                return 0

            return (piece_base_y - self.level) * (
                min(piece_end_x, self.end_x) - max(piece_start_x, self.start_x)
            )

    def initialize_height_profiles(self, plates):
        self.height_profiles = []
        self.plate_inventory = [(plate, i) for i, plate in enumerate(plates)]
        self.plate_inventory = sorted(
            self.plate_inventory, 
            key=lambda x: -(self._get_stock_size_(x[0])[0] * self._get_stock_size_(x[0])[1])
        )

        for plate, i in self.plate_inventory:
            self.height_profiles += [[
                Policy2352237.HeightSegment(0, self._get_stock_size_(plate)[0], 0)
            ]]

    def find_optimal_placement(self, piece, plate, plate_idx):
        plate_width, plate_height = self._get_stock_size_(plate)
        piece_width, piece_height = piece["size"]

        NO_SOLUTION = 1e18
        min_waste = NO_SOLUTION
        best_position = (-1, -1)
        should_rotate = False

        # Try left alignment
        for segment_idx, segment in enumerate(self.height_profiles[plate_idx]):
            pos_x, pos_y = segment.start_x, segment.level

            if not self._validate_placement(
                pos_x, pos_y, piece_width, piece_height, 
                plate_width, plate_height, plate, 
                self.height_profiles[plate_idx]
            ): 
                continue

            current_waste = sum(
                segment.compute_gap_area((piece_width, piece_height), (pos_x, pos_y))
                for segment in self.height_profiles[plate_idx]
            )

            if min_waste > current_waste:
                min_waste = current_waste
                best_position = (pos_x, pos_y)
                should_rotate = False

        # Try right alignment
        for segment_idx, segment in enumerate(self.height_profiles[plate_idx]):
            pos_x, pos_y = segment.end_x - piece_width, int(segment.level)

            if not self._validate_placement(
                pos_x, pos_y, piece_width, piece_height, 
                plate_width, plate_height, plate, 
                self.height_profiles[plate_idx]
            ): 
                continue

            current_waste = sum(
                segment.compute_gap_area((piece_width, piece_height), (pos_x, pos_y))
                for segment in self.height_profiles[plate_idx]
            )

            if min_waste > current_waste:
                min_waste = current_waste
                best_position = (pos_x, pos_y)
                should_rotate = False

        # Try rotated piece
        piece_width, piece_height = piece_height, piece_width

        # Repeat left and right alignment with rotated piece
        for segment_idx, segment in enumerate(self.height_profiles[plate_idx]):
            pos_x, pos_y = segment.start_x, segment.level

            if not self._validate_placement(
                pos_x, pos_y, piece_width, piece_height, 
                plate_width, plate_height, plate, 
                self.height_profiles[plate_idx]
            ): 
                continue

            current_waste = sum(
                segment.compute_gap_area((piece_width, piece_height), (pos_x, pos_y))
                for segment in self.height_profiles[plate_idx]
            )

            if min_waste > current_waste:
                min_waste = current_waste
                best_position = (pos_x, pos_y)
                should_rotate = True

        for segment_idx, segment in enumerate(self.height_profiles[plate_idx]):
            pos_x, pos_y = segment.end_x - piece_width, int(segment.level)

            if not self._validate_placement(
                pos_x, pos_y, piece_width, piece_height, 
                plate_width, plate_height, plate, 
                self.height_profiles[plate_idx]
            ): 
                continue

            current_waste = sum(
                segment.compute_gap_area((piece_width, piece_height), (pos_x, pos_y))
                for segment in self.height_profiles[plate_idx]
            )

            if min_waste > current_waste:
                min_waste = current_waste
                best_position = (pos_x, pos_y)
                should_rotate = True
        
        return int(min_waste), best_position, should_rotate

    def _validate_placement(self, x, y, width, height, plate_width, plate_height, plate, segments):
        if x < 0 or x + width > plate_width: return False
        if y < 0 or y + height > plate_height: return False
        if not self._can_place_(plate, (x, y), (width, height)): return False
        
        return not any(
            segment.has_overlap((width, height), (x, y))
            for segment in segments
        )

    def update_height_profile(self, piece, position, plate, plate_idx):
        updated_profile = []

        piece_width, piece_height = piece["size"]
        pos_x, pos_y = position
        start = int(pos_x)
        end = int(start + piece_width)
        new_height = pos_y + piece_height

        updated_profile.append(Policy2352237.HeightSegment(start, end, new_height))

        for segment in self.height_profiles[plate_idx]:
            if segment.end_x <= start or segment.start_x >= end:
                updated_profile.append(segment)
                continue
                
            if start <= segment.start_x and segment.end_x <= end:
                continue

            if segment.start_x < start:
                updated_profile.append(
                    Policy2352237.HeightSegment(segment.start_x, start, segment.level)
                )
            else:
                updated_profile.append(
                    Policy2352237.HeightSegment(end, segment.end_x, segment.level)
                )

        updated_profile.sort(key=lambda x: x.start_x)
        self.height_profiles[plate_idx] = updated_profile