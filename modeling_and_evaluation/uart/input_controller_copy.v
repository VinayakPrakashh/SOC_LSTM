module input_controller #(
    parameter OUTPUT_DATA_WIDTH = 80, // Address width for output data
    parameter ADDR_WIDTH = 5,         // Address width for FIFO
    parameter INPUT_DATA_WIDTH = 8                  // Data width for multiplier output
) (
    input clk,
    input start,
    input rst_n,
    input [INPUT_DATA_WIDTH-1:0] input_data,
    output reg pop_fifo,
    output [ADDR_WIDTH-1:0] output_addr,
    output reg [OUTPUT_DATA_WIDTH-1:0] output_data,
    output reg done,
    output reg wr_en

);



    // FSM States
    localparam IDLE = 3'd0;
    localparam POP_FIFO = 3'd1;
    localparam OUTPUT_READY = 3'd2;
    
    reg [2:0] state, next_state;

    reg [3:0] pop_counter;
    reg [79:0] combined_data; // Register to hold combined data
    reg pop_toggle; // Toggle to control when to read from FIFO\
    reg pop_fifo; // Register to hold pop signal for one cycle
    
    // FSM State Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // Next State Logic
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = POP_FIFO;
                else
                    next_state = IDLE;
            end
            
            POP_FIFO: begin
                if (pop_counter == 10) // After popping 10 times (0-9)
                    next_state = OUTPUT_READY;
                    else
                    next_state = POP_FIFO;
            end
            
            OUTPUT_READY: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end
    
    
    
    // Combine and Output Logic
    always @(posedge clk or negedge rst_n) begin
       if (!rst_n) begin
    output_data <= 0;
    done <= 0;
    pop_counter <= 0;
    wr_en <= 0;
    pop_fifo <= 0;
    pop_toggle <= 0;
end else begin
            case (state)
                IDLE: begin
                    output_data <= 0;
                    done <= 0;
                    pop_counter <= 0;
                    wr_en <= 0;
                    pop_fifo <= 0;
                end
                
POP_FIFO: begin

    pop_fifo <= pop_toggle;       // pulse every other clock
    pop_toggle <= ~pop_toggle;

    if(pop_toggle) begin          // increment only on pulse
        pop_counter <= pop_counter + 1;

        case(pop_counter)
            0: combined_data[7:0]   <= input_data;
            1: combined_data[15:8]  <= input_data;
            2: combined_data[23:16] <= input_data;
            3: combined_data[31:24] <= input_data;
            4: combined_data[39:32] <= input_data;
            5: combined_data[47:40] <= input_data;
            6: combined_data[55:48] <= input_data;
            7: combined_data[63:56] <= input_data;
            8: combined_data[71:64] <= input_data;
            9: combined_data[79:72] <= input_data;
        endcase

    end

end
                OUTPUT_READY: begin
                    output_data <= combined_data; // Output the combined data
                    wr_en <= 1; // Signal that output data is ready
                    done <= 1;
                end
                
                default: begin
                    done <= 1'b0;
                end
            endcase
        end
    end


endmodule
