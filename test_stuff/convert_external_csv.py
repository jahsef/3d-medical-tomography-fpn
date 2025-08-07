import pandas as pd
import ast

def convert_tomo_data(input_file, output_file):
    # Read the input data
    df = pd.read_csv(input_file, sep=',')
    
    output_rows = []
    row_id = 0
    
    for _, row in df.iterrows():
        # print(row)
        tomo_id = row['tomo_id']
        motor_z = row['motor_z'] if pd.notna(row['motor_z']) else -1
        motor_y = row['motor_y'] if pd.notna(row['motor_y']) else -1
        motor_x = row['motor_x'] if pd.notna(row['motor_x']) else -1
        voxel_spacing = row['voxel_spacing'] if pd.notna(row['voxel_spacing']) else -1
        num_motors = int(row['num_motors']) if pd.notna(row['num_motors']) else 0
        
        # Parse motor_zyx coordinates
        motor_zyx_str = row['motor_zyx']
        if pd.notna(motor_zyx_str) and motor_zyx_str != '[]':
            try:
                motor_coords = ast.literal_eval(motor_zyx_str)
                if not motor_coords:  # Empty list
                    motor_coords = []
            except:
                motor_coords = []
        else:
            motor_coords = []
        
        # If no motor coordinates, create one row with -1 values
        if not motor_coords:
            output_rows.append({
                'row_id': row_id,
                'tomo_id': tomo_id,
                'Motor axis 0': -1,
                'Motor axis 1': -1,
                'Motor axis 2': -1,
                'Array shape (axis 0)': motor_z,
                'Array shape (axis 1)': motor_y,
                'Array shape (axis 2)': motor_x,
                'Voxel spacing': voxel_spacing,
                'Number of motors': num_motors
            })
            row_id += 1
        else:
            # Create a row for each motor coordinate
            for coord in motor_coords:
                # Pad coordinate with -1 if less than 3 values
                while len(coord) < 3:
                    coord.append(-1)
                
                output_rows.append({
                    'row_id': row_id,
                    'tomo_id': tomo_id,
                    'Motor axis 0': coord[0],
                    'Motor axis 1': coord[1],
                    'Motor axis 2': coord[2],
                    'Array shape (axis 0)': motor_z,
                    'Array shape (axis 1)': motor_y,
                    'Array shape (axis 2)': motor_x,
                    'Voxel spacing': voxel_spacing,
                    'Number of motors': num_motors
                })
                row_id += 1
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Converted data saved to {output_file}")

# Usage
if __name__ == "__main__":
    convert_tomo_data(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\gt_v3.csv',
                      r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\relabel.csv')