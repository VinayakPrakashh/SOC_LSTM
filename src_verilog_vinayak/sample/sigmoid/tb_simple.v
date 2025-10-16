// Simple Testbench for Sigmoid Address Calculator (Sign Independent)
`timescale 1ns / 1ps

module tb_sigmoid_addr_calc_simple();

    // Test signals
    reg [11:0] data_in;
    wire [8:0] addr_out;
    wire out_of_range;
    
    // DUT instantiation
    sigmoid_addr_calc dut (
        .data_in(data_in),
        .addr_out(addr_out),
        .out_of_range(out_of_range)
    );
    
    // Test procedure
    initial begin
        $display("Sigmoid Address Calculator - Sign Independent Test");
        $display("Input\t\tMagnitude\tAddr\tOOR");
        $display("-----\t\t---------\t----\t---");
        
        // Test 1: Zero input
        data_in = 12'h000;  // 0.0
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 2: Small positive
        data_in = 12'h020;  // +0.5
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 3: Small negative (same magnitude as test 2)
        data_in = 12'h820;  // -0.5 (sign bit set, same magnitude)
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 4: Mid range positive
        data_in = 12'h0C0;  // +3.0
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 5: Mid range negative (same magnitude as test 4)
        data_in = 12'h8C0;  // -3.0 (sign bit set, same magnitude)
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 6: Maximum valid positive
        data_in = 12'h180;  // +6.0
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 7: Maximum valid negative (same magnitude as test 6)
        data_in = 12'h980;  // -6.0 (sign bit set, same magnitude)
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 8: Overflow positive
        data_in = 12'h200;  // +8.0
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        // Test 9: Overflow negative (same magnitude as test 8)
        data_in = 12'hA00;  // -8.0 (sign bit set, same magnitude)
        #10;
        $display("%h\t\t%h\t\t%d\t%b", data_in, data_in[10:0], addr_out, out_of_range);
        
        $display("");
        $display("Note: Address calculator now uses magnitude only.");
        $display("Positive and negative inputs with same magnitude give same address.");
        $display("Test completed");
        $finish;
    end
    
endmodule