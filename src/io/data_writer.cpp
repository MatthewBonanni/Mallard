/**
 * @file data_writer.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief DataWriter class implementation.
 * @version 0.1
 * @date 2024-01-01
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "data_writer.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <optional>

#include "common_io.h"

DataWriter::DataWriter() {
    // Empty
}

DataWriter::~DataWriter() {
    // Empty
}

void DataWriter::init(const toml::table & input,
                      std::vector<Data> & data,
                      std::shared_ptr<Mesh> mesh) {
    std::optional<std::string> prefix = input["prefix"].value<std::string>();
    std::optional<std::string> format_str = input["format"].value<std::string>();
    std::optional<int> interval = input["interval"].value<int>();
    auto variables = input["variables"];
    const toml::array* arr = variables.as_array();

    if (!prefix.has_value()) {
        throw std::runtime_error("DataWriter: prefix not specified.");
    }
    if (!format_str.has_value()) {
        throw std::runtime_error("DataWriter: format not specified.");
    }
    if (!interval.has_value()) {
        throw std::runtime_error("DataWriter: interval not specified.");
    }
    if (!variables) {
        throw std::runtime_error("DataWriter: variables not specified.");
    } else if (arr->size() == 0) {
        throw std::runtime_error("DataWriter: variables must be a non-empty array.");
    }

    this->prefix = prefix.value();
    this->interval = interval.value();

    DataFormat format;
    typename std::unordered_map<std::string, DataFormat>::const_iterator it = FORMAT_TYPES.find(*format_str);
    if (it == FORMAT_TYPES.end()) {
        throw std::runtime_error("DataWriter: Unknown format type: " + *format_str + ".");
    } else {
        format = it->second;
    }

    for (const auto & var : *arr) {
        std::optional<std::string> var_str = var.value<std::string>();
        if (!var_str.has_value()) {
            throw std::runtime_error("DataWriter: variable must be a string.");
        }

        bool found = false;
        for (const auto & data_var : data) {
            if (data_var.name() == var_str.value()) {
                data_ptrs.push_back(&data_var);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("DataWriter: Unknown variable: " + var_str.value() + ".");
        }
    }

    this->mesh = mesh;
}

void DataWriter::write(int step, bool force) const {
    bool write_now = (step % interval == 0) || force;
    if (!write_now) {
        return;
    }

    if (format == DataFormat::VTU) {
        write_vtu(step);
    } else if (format == DataFormat::TECPLOT) {
        write_tecplot(step);
    } else {
        throw std::runtime_error("DataWriter::write not implemented.");
    }
}

void DataWriter::write_vtu(int step) const {
    std::string filename;
    std::ostringstream stream;
    stream << std::setw(LEN_STEP) << std::setfill('0') << step;
    filename = prefix + "_" + stream.str() + ".vtu";

    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not open file: " + filename + ".");
    }
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not write to file: " + filename + ".");
    }

    std::cout << "Writing data to file: " << filename << std::endl;

    int offset_data = 0;
    int len_connectivity = 0;

    // ---------------------------------------------------------------------------------------------
    // Write header
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << endianness() << "\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh->n_nodes() << "\" NumberOfCells=\"" << mesh->n_cells() << "\">\n";

    // Write PointData
    out << "      <PointData>\n";
    out << "      </PointData>\n";

    // Write CellData
    out << "      <CellData>\n";
    for (const auto & data_ptr : data_ptrs) {
        out << "        <DataArray type=\"Float64\" Name=\"" << data_ptr->name() << "\" ";
        out << "format=\"appended\" ";
        out << "offset=\"" << offset_data << "\">\n";
        out << "        </DataArray>\n";
        offset_data += sizeof(int) + mesh->n_cells() * sizeof(double);
    }
    out << "      </CellData>\n";

    // Write Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"" << 3 << "\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    offset_data += sizeof(int) + mesh->n_nodes() * 3 * sizeof(double);
    out << "      </Points>\n";

    // Write Cells
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    for (int i = 0; i < mesh->n_cells(); i++) {
        len_connectivity += mesh->nodes_of_cell(i).size();
    }
    offset_data += sizeof(int) + len_connectivity * sizeof(int);

    // offsets
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    offset_data += sizeof(int) + mesh->n_cells() * sizeof(int);

    // types
    out << "        <DataArray type=\"Int32\" Name=\"types\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    offset_data += sizeof(int) + mesh->n_cells() * sizeof(int);

    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";

    // ---------------------------------------------------------------------------------------------
    // Write appended data
    out << "<AppendedData encoding=\"raw\">\n_";

    int n_bytes;

    // Write PointData
    // Do nothing, no point data

    // Write CellData
    for (const auto & data_ptr : data_ptrs) {
        int bytes = sizeof(double) * mesh->n_cells();
        out.write(reinterpret_cast<const char *>(&bytes), sizeof(int));
        for (int i = 0; i < mesh->n_cells(); i++) {
            out.write(reinterpret_cast<const char *>(&(*data_ptr)[i]), sizeof(double));
        }
    }

    // Write Points
    n_bytes = sizeof(double) * mesh->n_nodes() * 3;
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    for (int i = 0; i < mesh->n_nodes(); i++) {
        for (int j = 0; j < N_DIM; j++) {
            out.write(reinterpret_cast<const char *>(&mesh->node_coords(i)[j]), sizeof(double));
        }
        if (N_DIM == 2) {
            double zero = 0.0;
            out.write(reinterpret_cast<const char *>(&zero), sizeof(double));
        }
    }

    // Write Cells

    // connectivity
    n_bytes = sizeof(int) * len_connectivity;
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    for (int i = 0; i < mesh->n_cells(); i++) {
        for (int j = 0; j < mesh->nodes_of_cell(i).size(); j++) {
            out.write(reinterpret_cast<const char *>(&mesh->nodes_of_cell(i)[j]), sizeof(int));
        }
    }

    // // offsets
    n_bytes = sizeof(int) * mesh->n_cells();
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    int offset = 0;
    for (int i = 0; i < mesh->n_cells(); i++) {
        offset += mesh->nodes_of_cell(i).size();
        out.write(reinterpret_cast<const char *>(&offset), sizeof(int));
    }

    // // types
    n_bytes = sizeof(int) * mesh->n_cells();
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    int cell_type = 7;
    for (int i = 0; i < mesh->n_cells(); i++) {
        out.write(reinterpret_cast<const char *>(&cell_type), sizeof(int));
    }

    out << "\n";
    out << "</AppendedData>\n";
    out << "</VTKFile>\n";

    out.close();
}

void DataWriter::write_tecplot(int step) const {
    std::string filename = prefix + "_" + std::to_string(step) + ".dat";

    throw std::runtime_error("DataWriter::write_tecplot not implemented.");
}