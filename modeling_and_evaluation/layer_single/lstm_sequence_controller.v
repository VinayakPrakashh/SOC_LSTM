// LSTM Sequence Controller FSM - 20 Timesteps
module lstm_sequence_controller (
    input  clk,
    input  rst_n,
    input  start,              // Start signal
    input  [79:0] input_data,  // 5 inputs x 16 bits = 80 bits'
    output reg rd_input,
    input [15:0] weights,     // Weights for the LSTM (packed as needed)
    output reg[6:0] weight_addr,     // Address for weight memory (0-93)
    output reg [15:0] output_data, // Final output after FC layer
    output reg [6:0] addr_counter, // Address counter for input/weight loading
    output reg wr_en,              // Write enable for output data
    output reg done,               // Completion signal
    output reg busy                // FSM is busy
);

// ============================================================================
// FSM States
// ============================================================================

localparam [3:0] 
    IDLE           = 4'd0,
    INIT_LOAD     = 4'd1,
    SEQ_LOAD   = 4'd2,
    MULTIPLY  = 4'd3,
    UPDATE_CELL    = 4'd4,
    UPDATE_HIDDEN  = 4'd5,
    NEXT_TIMESTEP  = 4'd6,
    FC_LAYER       = 4'd7,
    OUTPUT_RESULT  = 4'd8,
    DONE           = 4'd9;

reg [3:0] state, next_state;
reg [4:0] timestep;  // 5 bits to count up to 20 timesteps
reg [79:0] current_input; // Register to hold current input data for processing

wire [15:0] input_mux_out; // Output from input multiplexer
// ============================================================================
// Timestep Counter
// ============================================================================

localparam MAX_TIMESTEPS = 20;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        timestep <= 5'd0;
    end else if (state == IDLE && start) begin
        timestep <= 5'd0;
    end else if (state == NEXT_TIMESTEP && timestep < MAX_TIMESTEPS - 1) begin
        timestep <= timestep + 1'b1;
    end else if (state == DONE) begin
        timestep <= 5'd0;
    end
end

// ============================================================================
// State Register
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state <= IDLE;
    else
        state <= next_state;
end

// ============================================================================
// Next State Logic
// ============================================================================

always @(*) begin
    next_state = state;
    
    case (state)
        IDLE: begin
            if (start)
                next_state = LOAD_INPUT;  // Start loading input data
        end
        LOAD_INPUT: begin
            next_state = LOAD;
        end
        LOAD: begin
            next_state = MULTIPLY;
        end
        
        MULTIPLY: begin
            next_state = UPDATE_CELL;
        end
        
        UPDATE_CELL: begin
            next_state = UPDATE_HIDDEN;
        end
        
        UPDATE_HIDDEN: begin
            next_state = NEXT_TIMESTEP;
        end
        
        NEXT_TIMESTEP: begin
            if (timestep < MAX_TIMESTEPS - 1)
                next_state = LOAD_INPUT;  // Go to next timestep
            else
                next_state = FC_LAYER;     // All timesteps done
        end
        
        FC_LAYER: begin
            next_state = OUTPUT_RESULT;
        end
        
        OUTPUT_RESULT: begin
            next_state = DONE;
        end
        
        DONE: begin
            next_state = IDLE;
        end
        
        default: begin
            next_state = IDLE;
        end
    endcase
end

// ============================================================================
// Output Logic
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 1'b0;
        busy <= 1'b0;
        timestep <= 5'd0;
        addr_counter <= 7'd0;
        rd_input <= 1'b0;
        weight_addr <= 7'd0;
        output_data <= 16'd0;
        wr_en <= 1'b0;
    end else begin
        case (state)
            IDLE: begin
                done <= 1'b0;
                busy <= 1'b0;
                timestep <= 5'd0;
                addr_counter <= 7'd0;
                output_data <= 16'd0;
                rd_input <= 1'b0;
                weight_addr <= 7'd0;
                wr_en <= 1'b0;

            end
            LOAD_INPUT: begin
                busy <= 1'b1;
                rd_input <= 1'b1;  // Signal to read input data
                // Addressing logic for input data can be implemented here
                current_input <= input_data; // Capture input data for processing
                

            end
            LOAD: begin
                busy <= 1'b1;
                wr_en <= 1'b1;  

                // Load initial input data for timestep 0
                addr_counter <= addr_counter + 1;
                if(addr_counter < 5) begin
                    output_data <= input_mux_out; // Capture input data for processing
                end else if((addr_counter > 4) && (addr_counter < 99)) begin
                    weight_addr <= weight_addr + 1; // Increment weight address for loading
                    output_data <= (timestep == 0)? 0:weights; // Capture weight data for processing
                end
                else if(addr_counter == 99) begin
                    output_data <= 1; // Capture last weight data for processing
                end
                
            end
            
            OUTPUT_RESULT: begin
                // Final output calculated here
                done <= 1'b0;
            end
            
            DONE: begin
                done <= 1'b1;
                busy <= 1'b0;
            end
            
            default: begin
                busy <= 1'b1;
                done <= 1'b0;
            end
        endcase
    end
end

// ============================================================================
// Hidden State and Cell State Storage (94 units each)
// ============================================================================

reg [15:0] h_state [0:93];  // Hidden state (94 units)
reg [15:0] c_state [0:93];  // Cell state (94 units)

integer i;

// Initialize states to zero at start
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 94; i = i + 1) begin
            h_state[i] <= 16'd0;
            c_state[i] <= 16'd0;
        end
    end else if (state == IDLE && start) begin
        // Reset states for new sequence
        for (i = 0; i < 94; i = i + 1) begin
            h_state[i] <= 16'd0;
            c_state[i] <= 16'd0;
        end
    end
end

// ============================================================================
// Cycle Counter (for debugging/monitoring)
// ============================================================================

reg [15:0] cycle_count;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycle_count <= 16'd0;
    end else if (state == IDLE) begin
        cycle_count <= 16'd0;
    end else if (busy) begin
        cycle_count <= cycle_count + 1'b1;
    end
end
mux_5to1 #(
    .WIDTH(16)
) input_mux (
    .in0(current_input[15:0]),   // Input 0
    .in1(current_input[31:16]),  // Input 1
    .in2(current_input[47:32]),  // Input 2
    .in3(current_input[63:48]),  // Input 3
    .in4(current_input[79:64]),  // Input 4
    .sel(addr_counter[2:0]),     // Select based on address counter (modulo 5)
    .out(input_mux_out)               // Output to weight multiplier
);

endmodule