`timescale 1ns /1ps

module fifo
    #(
       parameter	DATA_SIZE 	   = 8,
                    ADDR_SPACE_EXP = 4,
                    FIFO_DEPTH     = 9        // <-- actual number of locations
    )
    (
       input clk,
       input reset,
       input write_to_fifo,
       input read_from_fifo,
       input [DATA_SIZE-1:0] write_data_in,
       output [DATA_SIZE-1:0] read_data_out,
       output empty,
       output full
);

    // signal declaration
    reg [DATA_SIZE-1:0] memory [FIFO_DEPTH-1:0];   // exactly 10 locations
    reg [ADDR_SPACE_EXP-1:0] current_write_addr, current_write_addr_buff, next_write_addr;
    reg [ADDR_SPACE_EXP-1:0] current_read_addr, current_read_addr_buff, next_read_addr;
    reg fifo_full, fifo_empty, full_buff, empty_buff;
    wire write_enabled;
    
    // register file (memory) write operation
    always @(posedge clk)
        if(write_enabled)
            memory[current_write_addr] <= write_data_in;
            
    // register file (memory) read operation
    assign read_data_out = memory[current_read_addr];
    
    // only allow write operation when FIFO is NOT full
    assign write_enabled = write_to_fifo & ~fifo_full;
    
    // FIFO control logic
    always @(posedge clk or posedge reset)
        if(reset) begin
            current_write_addr 	<= 0;
            current_read_addr 	<= 0;
            fifo_full 			<= 1'b0;
            fifo_empty 			<= 1'b1;
        end
        else begin
            current_write_addr  <= current_write_addr_buff;
            current_read_addr   <= current_read_addr_buff;
            fifo_full  			<= full_buff;
            fifo_empty 			<= empty_buff;
        end

    // next state logic - wrap at FIFO_DEPTH instead of power of 2
    always @* begin
        // wrap pointers at FIFO_DEPTH (10)
        next_write_addr = (current_write_addr == FIFO_DEPTH-1) ? 0 : current_write_addr + 1;
        next_read_addr  = (current_read_addr  == FIFO_DEPTH-1) ? 0 : current_read_addr  + 1;
        
        // default: keep old values
        current_write_addr_buff = current_write_addr;
        current_read_addr_buff  = current_read_addr;
        full_buff  = fifo_full;
        empty_buff = fifo_empty;
        
        case({write_to_fifo, read_from_fifo})
            2'b01:	// read
                if(~fifo_empty) begin
                    current_read_addr_buff = next_read_addr;
                    full_buff = 1'b0;
                    if(next_read_addr == current_write_addr)
                        empty_buff = 1'b1;
                end
            
            2'b10:	// write
                if(~fifo_full) begin
                    current_write_addr_buff = next_write_addr;
                    empty_buff = 1'b0;
                    if(next_write_addr == current_read_addr)
                        full_buff = 1'b1;
                end
                
            2'b11:	begin	// simultaneous read and write
                current_write_addr_buff = next_write_addr;
                current_read_addr_buff  = next_read_addr;
                end
        endcase			
    end

    assign full  = fifo_full;
    assign empty = fifo_empty;

endmodule